#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>


// #define DEBUG

// Kernel: One block per query; each thread computes local top-k, then merge using shared memory.
template <typename scalar_t>
__global__ void knn_cuda_kernel(
    const scalar_t* __restrict__ query,    // (M, D)
    const scalar_t* __restrict__ database, // (N, D)
    const int M,  // number of query points
    const int N,  // number of database points
    const int D,  // feature dimension
    const int64_t k,  // number of neighbors to find
    scalar_t* __restrict__ distances,      // output: (M, k)
    int64_t* __restrict__ indices          // output: (M, k)
) {
    int query_idx = blockIdx.y * gridDim.x + blockIdx.x;
    if (query_idx >= M) return;

    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    int K = static_cast<int>(k);

    const scalar_t* q = query + query_idx * D;

    // Each thread keeps a local top-k list.
    scalar_t local_dist[256];
    int local_idx[256];
    for (int i = 0; i < K; i++){
        local_dist[i] = 1e20;
        local_idx[i] = -1;
    }

#ifdef DEBUG
    if (query_idx == 0 && tid == 0)
        printf("Query 0 first feature: %f\n", q[0]);
#endif

    // Loop over database points in a strided manner.
    for (int j = tid; j < N; j += stride) {
        scalar_t dist = 0;
        for (int d = 0; d < D; d++){
            scalar_t diff = q[d] - database[j * D + d];
            dist += diff * diff;
        }
        int max_pos = 0;
        scalar_t max_val = local_dist[0];
        for (int i = 1; i < K; i++){
            if (local_dist[i] > max_val){
                max_val = local_dist[i];
                max_pos = i;
            }
        }
        if (dist < max_val) {
            local_dist[max_pos] = dist;
            local_idx[max_pos] = j;
        }
    }

#ifdef DEBUG
    if(query_idx == 0 && tid == 0){
        printf("Thread %d local top-k: ", tid);
        for (int i = 0; i < K; i++){
            printf("%f ", local_dist[i]);
        }
        printf("\n");
    }
#endif

    // Shared memory for merging.
    extern __shared__ char shared_memory[];
    scalar_t* s_dist = reinterpret_cast<scalar_t*>(shared_memory);
    int64_t* s_idx = reinterpret_cast<int64_t*>(s_dist + blockDim.x * K);

    for (int i = 0; i < K; i++){
        s_dist[tid * K + i] = local_dist[i];
        s_idx[tid * K + i] = local_idx[i];
    }
    __syncthreads();

#ifdef DEBUG
    if(query_idx == 0 && tid == 0){
        printf("Shared memory (first 10): ");
        for (int j = 0; j < min(blockDim.x * K, 10); j++){
            printf("%f ", s_dist[j]);
        }
        printf("\n");
    }
#endif

    // Merge the local top-k lists.
    if(tid == 0){
        int total = blockDim.x * K;
        for (int i = 0; i < K; i++){
            scalar_t best = 1e20;
            int best_index = -1;
            int best_pos = -1;
            for (int j = 0; j < total; j++){
                scalar_t val = s_dist[j];
                if (val < best){
                    best = val;
                    best_index = s_idx[j];
                    best_pos = j;
                }
            }
            distances[query_idx * K + i] = best;
            indices[query_idx * K + i] = best_index;
            s_dist[best_pos] = 1e20; // mark as used
        }
#ifdef DEBUG
        if(query_idx == 0){
            printf("Merged distances for query 0: ");
            for (int i = 0; i < K; i++){
                printf("%f ", distances[query_idx * K + i]);
            }
            printf("\n");
        }
#endif
    }
}

std::tuple<torch::Tensor, torch::Tensor> knn_cuda(torch::Tensor query, torch::Tensor database, int64_t k) {
    const int M = query.size(0);
    const int N = database.size(0);
    const int D = query.size(1);

    auto distances = torch::empty({M, k}, query.options());
    auto indices = torch::empty({M, k}, torch::dtype(torch::kInt64).device(query.device()));

    // If k is large, reduce threads per block to keep shared memory within limits.
    int threads = (k > 15) ? 32 : 256;
    // Use a 2D grid in case M is very large.
    int maxBlocks = 65535;
    int grid_x = (M < maxBlocks ? M : maxBlocks);
    int grid_y = (M + maxBlocks - 1) / maxBlocks;
    dim3 blocks(grid_x, grid_y);
    size_t shared_mem_size = threads * k * (sizeof(float) + sizeof(int64_t));

    AT_DISPATCH_FLOATING_TYPES(query.scalar_type(), "knn_cuda", ([&] {
        knn_cuda_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
            query.data_ptr<scalar_t>(),
            database.data_ptr<scalar_t>(),
            M, N, D, k,
            distances.data_ptr<scalar_t>(),
            indices.data_ptr<int64_t>());
    }));
    cudaDeviceSynchronize();
    return std::make_tuple(distances, indices);
}

TORCH_LIBRARY(knn_cuda, m) {
    m.def("knn_cuda(Tensor query, Tensor database, int k) -> (Tensor, Tensor)");
    m.impl("knn_cuda", knn_cuda);
}
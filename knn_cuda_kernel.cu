#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

// #define DEBUG


// Kernel: Each block processes one query.
// Each thread computes its local top-k (stored in registers), writes them into shared memory,
// then all threads collaborate to iteratively extract the k smallest candidates.
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
    // Compute global query index from a 2D grid.
    int query_idx = blockIdx.y * gridDim.x + blockIdx.x;
    if (query_idx >= M) return;

    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int stride = blockSize;
    int K = static_cast<int>(k);

    // Pointer to the current query vector.
    const scalar_t* q = query + query_idx * D;

    // Each thread maintains a local top-k list.
    scalar_t local_dist[128];
    int local_idx[128];
    for (int i = 0; i < K; i++) {
        local_dist[i] = 1e20;
        local_idx[i] = -1;
    }

#ifdef DEBUG
    if (query_idx == 0 && tid == 0)
        printf("Query 0 first feature: %f\n", q[0]);
#endif

    // Each thread processes its subset of the database.
    for (int j = tid; j < N; j += stride) {
        scalar_t dist = 0;
        for (int d = 0; d < D; d++) {
            scalar_t diff = q[d] - database[j * D + d];
            dist += diff * diff;
        }
        // Find the maximum value in the thread's local top-k list.
        int max_pos = 0;
        scalar_t max_val = local_dist[0];
        for (int i = 1; i < K; i++) {
            if (local_dist[i] > max_val) {
                max_val = local_dist[i];
                max_pos = i;
            }
        }
        // If the current distance is smaller than the worst candidate, update it.
        if (dist < max_val) {
            local_dist[max_pos] = dist;
            local_idx[max_pos] = j;
        }
    }

    // Allocate shared memory to hold each thread's local k candidates.
    // Shared memory layout: first blockSize*K elements for distances, next blockSize*K for indices.
    extern __shared__ char shared_memory[];
    scalar_t* s_dist = reinterpret_cast<scalar_t*>(shared_memory);
    int* s_idx = reinterpret_cast<int*>(s_dist + blockSize * K);

    // Write each thread's local top-k into shared memory.
    for (int i = 0; i < K; i++) {
        s_dist[tid * K + i] = local_dist[i];
        s_idx[tid * K + i] = local_idx[i];
    }
    __syncthreads();

#ifdef DEBUG
    if (query_idx == 0 && tid == 0) {
        printf("Shared memory (first 10 values): ");
        for (int j = 0; j < min(blockSize * K, 10); j++){
            printf("%f ", s_dist[j]);
        }
        printf("\n");
    }
#endif

    // Iterative merge: all threads participate to extract k best candidates.
    int total = blockSize * K;
    for (int iter = 0; iter < K; iter++) {
        // Each thread computes a candidate minimum over its assigned indices.
        scalar_t thread_best = 1e20;
        int thread_best_pos = -1;
        for (int j = tid; j < total; j += blockSize) {
            scalar_t val = s_dist[j];
            if (val < thread_best) {
                thread_best = val;
                thread_best_pos = j;
            }
        }
        // Now perform a warp-level reduction over all threads in the block.
        unsigned int mask = 0xffffffff;
        for (int offset = blockSize / 2; offset > 0; offset /= 2) {
            scalar_t other_val = __shfl_down_sync(mask, thread_best, offset, blockSize);
            int other_pos = __shfl_down_sync(mask, thread_best_pos, offset, blockSize);
            if (other_val < thread_best) {
                thread_best = other_val;
                thread_best_pos = other_pos;
            }
        }
        // The thread with tid == 0 writes the candidate for this iteration.
        if (tid == 0) {
            distances[query_idx * K + iter] = thread_best;
            // Use the index from shared memory.
            indices[query_idx * K + iter] = s_idx[thread_best_pos];
        }
        __syncthreads();
        // One thread (tid==0) marks the found candidate as used.
        if (tid == 0) {
            s_dist[thread_best_pos] = 1e20;
        }
        __syncthreads();
    }
}

std::tuple<torch::Tensor, torch::Tensor> knn_cuda(torch::Tensor query, torch::Tensor database, int64_t k) {
    const int M = query.size(0);
    const int N = database.size(0);
    const int D = query.size(1);

    auto distances = torch::empty({M, k}, query.options());
    auto indices = torch::empty({M, k}, torch::dtype(torch::kInt64).device(query.device()));

    // Choose block size: if k > 15, reduce threads per block.
    int threads = (k > 15) ? 32 : 128;
    // 2D grid for large M.
    int maxBlocks = 65535;
    int grid_x = (M < maxBlocks ? M : maxBlocks);
    int grid_y = (M + maxBlocks - 1) / maxBlocks;
    dim3 blocks(grid_x, grid_y);
    // Shared memory: need blockSize*K floats and blockSize*K ints.
    size_t shared_mem_size = threads * k * (sizeof(float) + sizeof(int));
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
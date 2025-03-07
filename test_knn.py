#!/usr/bin/env python3
import torch
import time
from typing import Tuple
from fastgraphcompute import binned_select_knn

torch.ops.load_library("./knn_cuda.so")
print("Registered knn_cuda operator:", torch.ops.knn_cuda)

@torch.jit.script
def knn_cuda_wrapper(query: torch.Tensor, database: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.knn_cuda.knn_cuda(query, database, k)


def knn_fastgraphcompute(x: torch.Tensor, k: int)-> Tuple[torch.Tensor, torch.Tensor]:
    row_splits = torch.tensor([0, x.shape[0]], dtype=torch.int32, device=x.device)
    neighbor_idx, distsq = binned_select_knn(k, x, row_splits)
    return distsq, neighbor_idx

def benchmark(func, *args, **kwargs):
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())
        torch.cuda.synchronize()
    start = time.time()
    result = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    mem = None
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / (1024**2)
        print(f"Memory: {mem:.2f} MB", end="; ")
    return result, elapsed

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    N = 15000
    D = 64
    k = 100

    query = torch.randn(N, D, device=device)
    database = query

    for _ in range(3):
        knn_cuda_wrapper(query, database, k)
        knn_fastgraphcompute(query, k)
        if device == "cuda":
            torch.cuda.synchronize()


    '''query = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [0.0, 2.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [1.0, 3.0],
    ], dtype=torch.float, device=device)
    database = query
    k = 5'''
    
    # For Custom CUDA kNN Operator:
    print("\n--- Custom CUDA kNN Operator ---")
    (distances_cuda, indices_cuda), time_cuda = benchmark(knn_cuda_wrapper, query, database, k)
    print(f"\nknn_cuda -- Time: {time_cuda:.4f} s")
    print("  Distances shape:", distances_cuda.shape)

    
    sorted_distances_cuda, sort_idx_cuda = torch.sort(distances_cuda[0])
    sorted_indices_cuda = indices_cuda[0][sort_idx_cuda]
    print("\n--- Sorted Results for First Query (Custom CUDA) ---")
    print("Sorted Distances:", sorted_distances_cuda.cpu())
    print("Sorted Indices:  ", sorted_indices_cuda.cpu())


    # For FastGraph:
    print("\n--- FastGraph ---")
    (distances_fg, indices_fg), time_fg = benchmark(knn_fastgraphcompute, query, k)
    print(f"\nFastGraph -- Time: {time_fg:.4f} s")
    print("  Distances shape:", distances_fg.shape)

    sorted_distances_fg, sort_idx_fg = torch.sort(distances_fg[0])
    sorted_indices_fg = indices_fg[0][sort_idx_fg]
    print("\n--- Sorted Results for First Query (FastGraph) ---")
    print("Sorted Distances:", sorted_distances_fg.cpu())
    print("Sorted Indices:  ", sorted_indices_fg.cpu())

   
    print("\n--- TorchScript Compatibility Check ---")
    try:
        torch.jit.save(knn_cuda_wrapper, "knn_cuda.pt")
        loaded_knn = torch.jit.load("knn_cuda.pt")
        print("knn_cuda operator: TorchScript Compatible")
        (distances_loaded, indices_loaded), time_loaded = benchmark(loaded_knn, query, database, k)
        print(f"\nLoaded knn_cuda -- Time: {time_loaded:.4f} s")
        print("Loaded knn_cuda distances (first query):")
        print(distances_loaded[:1].cpu())
    except Exception as e:
        print("knn_cuda operator: TorchScript Incompatible", e)



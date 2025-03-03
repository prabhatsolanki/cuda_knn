#!/usr/bin/env python3
import torch
import time
from typing import Tuple
from pykeops.torch import LazyTensor

torch.ops.load_library("./knn_cuda.so")
print("Registered knn_cuda operator:", torch.ops.knn_cuda)

@torch.jit.script
def knn_cuda_wrapper(query: torch.Tensor, database: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.knn_cuda.knn_cuda(query, database, k)


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

    N = 7000
    D = 32
    k = 80

    query = torch.randn(N, D, device=device)
    database = query

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
    k = 10'''
    
    print("\n--- Custom CUDA kNN Operator ---")
    (distances_cuda, indices_cuda), time_cuda = benchmark(knn_cuda_wrapper, query, database, k)
    print(f"\nknn_cuda -- Time: {time_cuda:.4f} s")
    print("  Distances shape:", distances_cuda.shape)


   
    #print("\n--- Example Distances (first 5 rows) ---")
    #print("Custom CUDA kNN distances (CPU):")
    #print(distances_cuda[:5].cpu(),indices_cuda[:5].cpu() )

   
    print("\n--- TorchScript Compatibility Check ---")
    try:
        torch.jit.save(knn_cuda_wrapper, "knn_cuda.pt")
        loaded_knn = torch.jit.load("knn_cuda.pt")
        print("knn_cuda operator: TorchScript Compatible")
        (distances_loaded, indices_loaded), time_loaded = benchmark(loaded_knn, query, database, k)
        print(f"\nLoaded knn_cuda -- Time: {time_loaded:.4f} s")
        #print("Loaded knn_cuda distances (first 5 rows):")
        #print(distances_loaded[:5].cpu())
    except Exception as e:
        print("knn_cuda operator: TorchScript Incompatible", e)
#include <stdio.h>

__device__ int arrivedThreads = 0;

__global__ void synchronizationDemo()
{
    int tid = threadIdx.x;

    printf("[Thread %d] Arrived at synchronization checkpoint\n", tid);

    atomicAdd(&arrivedThreads, 1);

    while (arrivedThreads < blockDim.x);

    __syncthreads();

    printf("[Thread %d] Continuing execution after synchronization\n", tid);
}

int main()
{
    synchronizationDemo<<<1, 5>>>();

    cudaDeviceSynchronize();

    printf("\nKernel execution completed successfully.\n");

    return 0;
}
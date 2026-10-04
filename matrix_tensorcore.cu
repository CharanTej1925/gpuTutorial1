#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define N 64
#define TILE 16

// One warp computes one 16x16 tile of C
__global__ void matrixMulTensorCore(const half *A,
                                    const half *B,
                                    float *C)
{
    // Each block contains one warp
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE,
                   half, wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE,
                   half, wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, TILE, TILE, TILE,
                   float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Multiply four 16x16 tiles along the K dimension
    for (int k = 0; k < N; k += TILE)
    {
        const half *A_tile =
            A + tileRow * TILE * N + k;

        const half *B_tile =
            B + k * N + tileCol * TILE;

        // Load 16x16 tiles into Tensor Core fragments
        wmma::load_matrix_sync(a_frag, A_tile, N);
        wmma::load_matrix_sync(b_frag, B_tile, N);

        // Tensor Core matrix multiplication
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the resulting 16x16 tile
    float *C_tile =
        C + tileRow * TILE * N + tileCol * TILE;

    wmma::store_matrix_sync(C_tile, c_frag, N,
                            wmma::mem_row_major);
}


int main()
{
    half *h_A;
    half *h_B;
    float *h_C;
    float *h_C_reference;

    // Allocate host memory
    h_A = new half[N * N];
    h_B = new half[N * N];
    h_C = new float[N * N];
    h_C_reference = new float[N * N];

    // Initialize matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] =
                __float2half((float)((i + j) % 5));

            h_B[i * N + j] =
                __float2half((float)((i - j + 64) % 5));

            h_C[i * N + j] = 0.0f;
            h_C_reference[i * N + j] = 0.0f;
        }
    }

    // CPU reference matrix multiplication
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < N; k++)
            {
                sum += __half2float(h_A[i * N + k]) *
                       __half2float(h_B[k * N + j]);
            }

            h_C_reference[i * N + j] = sum;
        }
    }

    // Device memory
    half *d_A;
    half *d_B;
    float *d_C;

    cudaMalloc((void **)&d_A, N * N * sizeof(half));
    cudaMalloc((void **)&d_B, N * N * sizeof(half));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Copy input matrices to GPU
    cudaMemcpy(d_A, h_A,
               N * N * sizeof(half),
               cudaMemcpyHostToDevice);

    cudaMemcpy(d_B, h_B,
               N * N * sizeof(half),
               cudaMemcpyHostToDevice);

    cudaMemset(d_C, 0, N * N * sizeof(float));

    /*
       64x64 matrix divided into 16x16 tiles:

              4 tiles
       +----+----+----+----+
       |  0 |  1 |  2 |  3 |
       +----+----+----+----+
       |  4 |  5 |  6 |  7 |
       +----+----+----+----+
       |  8 |  9 | 10 | 11 |
       +----+----+----+----+
       | 12 | 13 | 14 | 15 |
       +----+----+----+----+

       Total = 16 output tiles.

       Each block contains one warp (32 threads)
       and computes one 16x16 output tile.
    */

    dim3 blockDim(32);
    dim3 gridDim(N / TILE, N / TILE);

    // Launch Tensor Core kernel
    matrixMulTensorCore<<<gridDim, blockDim>>>(d_A, d_B, d_C);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C,
               N * N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Verify result
    bool correct = true;
    float maxError = 0.0f;

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float error =
                fabs(h_C[i * N + j] -
                     h_C_reference[i * N + j]);

            if (error > maxError)
                maxError = error;

            if (error > 0.1f)
                correct = false;
        }
    }

    printf("Matrix size : %dx%d\n", N, N);
    printf("Tile size   : %dx%d\n", TILE, TILE);
    printf("Output tiles: %dx%d = %d\n",
           N / TILE,
           N / TILE,
           (N / TILE) * (N / TILE));

    printf("Max error   : %f\n", maxError);

    if (correct)
        printf("Result      : CORRECT\n");
    else
        printf("Result      : INCORRECT\n");

    // Print a small part of the result
    printf("\nFirst 4x4 elements of C:\n");

    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            printf("%8.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_reference;

    return 0;
}

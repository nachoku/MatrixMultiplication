#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// // CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// // CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#define threadX 32
#define threadY 32
typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        // data[i]=float(i);
        // printf("%f ", data[i]);
        data[i] = rand() / (float)RAND_MAX;
      }
        
    // printf("\n");
}

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            double sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += a * b;
            }

            C[i * wB + j] = (float)sum;
        }
}

void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;

    for (j = 0; j < height; j++)
    {
        if (error_count < iListLength)
        {
            // printf("\n  Row %d:\n", j);
        }

        for (i = 0; i < width; i++)
        {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);

            if (fDiff > fListTol)
            {
                if (error_count < iListLength)
                {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }

                error_count++;
            }
        }
    }

    printf(" \n  Total Errors = %d\n", error_count);
}

__global__
void element(int n, int row_num_a, int col_num_a, int row_num_b, int col_num_b, int row_num_c, int col_num_c, float *a, float *b, float *c)
{
  __shared__ float A[threadX][threadY];//2D array for storing shared matrix values of A & B
  __shared__ float B[threadX][threadY];//Process subMatrix in block

  
  int Col=blockIdx.x*blockDim.x+threadIdx.x;//Col and Row Ids of threads
  int Row=blockIdx.y*blockDim.y+threadIdx.y;
  double temp = 0;
  for (int i = 0; i < (col_num_a-1)/blockDim.x+1; ++i) 
  {
     if (Row < row_num_a && i*blockDim.x+threadIdx.x < col_num_a)
        A[threadIdx.y][threadIdx.x] = a[Row*col_num_a + i*blockDim.x+threadIdx.x];//Memory Fetch from a
     else
        A[threadIdx.y][threadIdx.x] = 0;//In case the block dim is not a multiple of matrix

     if (Col < col_num_b && i*blockDim.x+threadIdx.y < row_num_b)
        B[threadIdx.y][threadIdx.x] = b[(i*blockDim.x+threadIdx.y)*col_num_b+Col];//Memory Fetch from b
     else
        B[threadIdx.y][threadIdx.x] = 0;

     __syncthreads();//Wait for all matrix loads to shared memory - then proceed with for loop
      if (Row < row_num_c && Col < col_num_c)
         
      for (int j = 0; j < blockDim.x; ++j)//Matrix multiplication
              temp += A[threadIdx.y][j] * B[j][threadIdx.x];
     __syncthreads();
  }
    if(Row<row_num_c && Col<col_num_c)//If the matrix is needed, then do this
       c[Row*col_num_c+Col] = (float)temp;//Save to c
  
  
}

int main(void)
{
  int nIter = 30;
  //Set Matrix Sizes
  sMatrixSize matrix_size;
  int mul=20;
  matrix_size.uiWA = mul* 160;
  matrix_size.uiHA = mul* 160;
  matrix_size.uiWB = mul* 160;
  matrix_size.uiHB = mul* 160;
  matrix_size.uiWC = mul* 160;
  matrix_size.uiHC = mul* 160;

  printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           matrix_size.uiHA, matrix_size.uiWA,
           matrix_size.uiHB, matrix_size.uiWB,
           matrix_size.uiHC, matrix_size.uiWC);

    if( matrix_size.uiWA != matrix_size.uiHB ||
        matrix_size.uiHA != matrix_size.uiHC ||
        matrix_size.uiWB != matrix_size.uiWC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
  //Number of Elements
  unsigned int size_A = matrix_size.uiWA * matrix_size.uiHA;
  unsigned int size_B = matrix_size.uiWB * matrix_size.uiHB;
  unsigned int size_C = matrix_size.uiWC * matrix_size.uiHC;
  // printf("Size of A: %d \n", size_A);
  // printf("Size of B: %d \n", size_B);
  //Memory Size
  unsigned int mem_size_A = sizeof(float) * size_A;
  unsigned int mem_size_B = sizeof(float) * size_B;
  unsigned int mem_size_C = sizeof(float) * size_C;

  //Initialize pointer variables
  float *a, *b, *c;

  //Allocate Space for Matrix on Host & Device- Pointer Variables
  cudaMallocManaged(&a, mem_size_A);
  cudaMallocManaged(&b, mem_size_B);
  cudaMallocManaged(&c, mem_size_C);

  //Fill Elements
  randomInit(a, size_A);
  randomInit(b, size_B);  


  unsigned int N=size_C;

  // Perform Matrix Multiplication & Record
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Warmup Execution
  printf("Block Dimensions: %dx%d\n",threadX, threadY);  
  dim3 threads(threadX, threadY);
  dim3 grid(matrix_size.uiWC/threads.x+1, matrix_size.uiHC/threads.y+1);
element<<<grid, threads>>>(N, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiHB, matrix_size.uiWB, matrix_size.uiHC, matrix_size.uiWC, a, b, c);

  //Actual execution
  cudaEventRecord(start, NULL);
  for (int j = 0; j < nIter; j++)
  {
  element<<<grid, threads>>>(N, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiHB, matrix_size.uiWB, matrix_size.uiHC, matrix_size.uiWC, a, b, c);
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * (double)matrix_size.uiHC * (double)matrix_size.uiWC * (double)matrix_size.uiHB;
  double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  
  printf( "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

  //Wait for GPU Finish
  cudaDeviceSynchronize();


  float *reference = (float *)malloc(mem_size_C);
  matrixMulCPU(reference, a, b, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);

bool resCUBLAS = sdkCompareL2fe(reference, c, size_C, 1.0e-6f);

    if (resCUBLAS != true)
    {
  printDiff(reference, c, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-4f);
    }

    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");

  //Release Resources
  // cudaFree(d_a);
  // cudaFree(d_b);
  // cudaFree(d_c);
  
  // free(a);
  // free(b);
  // free(c);
  cudaDeviceReset();
}

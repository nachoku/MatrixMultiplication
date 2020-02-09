////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
#include <cuda_runtime.h>
#include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
#include <helper_cuda.h>
#include <omp.h>

#ifndef min
#define min(a,b) ((a < b) ? a : b)
#endif
#ifndef max
#define max(a,b) ((a > b) ? a : b)
#endif

typedef struct _matrixSize      // Optional Command-line multiplier for matrix sizes
{
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
} sMatrixSize;

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set matrix multiply on CPU
//! C = A * B
//! @param C          reference data, computed but preallocated
//! @param A          matrix A as provided to device
//! @param B          matrix B as provided to device
//! @param hA         height of matrix A
//! @param wB         width of matrix B
////////////////////////////////////////////////////////////////////////////////
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
            // printf("%f\n", sum);
            C[i * wB + j] = (float)sum;
        }

}

void gemm_omp(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{

  // #pragma omp target teams distribute 
//#pragma omp target teams distribute
//#pragma omp target
    
#pragma omp target teams distribute 
   {
#pragma omp parallel for
      for (int i = 0; i < hA; ++i)
      {
#pragma omp parallel for
        for (int j = 0; j < wB; ++j)
        {
            double sum = 0;
            for (int k = 0; k < wA; ++k)
            {
                double a = A[i * wA + k];
                double b = B[k * wB + j];
                sum += A[i*wA+k] * B[k*wB+j];
	    }
            C[i * wB + j] = (float)sum;
        }
     }
    
  }
}

// Allocates a matrix with random float entries.
void randomInit(float *data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
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
            printf("\n  Row %d:\n", j);
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



int main(void)
{
  int nIter = 30;
  //Set Matrix Sizes
  int si=1280;
  sMatrixSize matrix_size;
  matrix_size.uiWA =  si;
  matrix_size.uiHA =  si;
  matrix_size.uiWB =  si;
  matrix_size.uiHB =  si;
  matrix_size.uiWC =  si;
  matrix_size.uiHC =  si;

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
  a = (float *)malloc(mem_size_A);
  b = (float *)malloc(mem_size_B);
  c = (float *)malloc(mem_size_C);
  //Fill Elements
  randomInit(a, size_A);
  randomInit(b, size_B);  


  unsigned int N=size_C;

  // Perform Matrix Multiplication & Record
  cublasHandle_t handle;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  //Warmup Execution
  cudaDeviceProp deviceProp;
  int block_size = (deviceProp.major < 2) ? 16 : 32;

  dim3 threads(block_size, block_size);
  dim3 blocks(matrix_size.uiWC / threads.x, matrix_size.uiHC / threads.y);

  gemm_omp(c, a, b, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
  // element<<<blocks, threads>>>(N, matrix_size.uiHB, a, b, c);

  //Actual execution
  cudaEventRecord(start, NULL);
  for (int j = 0; j < nIter; j++)
  {
        gemm_omp(c, a, b, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
    // element<<<blocks, threads>>>(N, matrix_size.uiHB, d_a, d_b, d_c);
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




// compute reference solution
  printf("Computing result using host CPU...");
  float *reference = (float *)malloc(mem_size_C);
  matrixMulCPU(reference, a, b, matrix_size.uiHA, matrix_size.uiWA, matrix_size.uiWB);
  printf("done.\n");

// check result (CUBLAS)
  
  float *h_CUBLAS = (float *) malloc(mem_size_C);
  bool resCUBLAS = sdkCompareL2fe(reference, c, size_C, 1.0e-6f);

  if (resCUBLAS != true)
  {
    printDiff(reference, c, matrix_size.uiWC, matrix_size.uiHC, 100, 1.0e-5f);
  }
  printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
                                                 
  cudaDeviceReset();
}

// saxpy.cu: Example program for CUDA programming

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <cstdlib>


__device__ __noinline__
void DataCorruption_Handler() {
    printf("Errore ASPIS: Data corruption rilevata\n");
    asm("trap;");
}
__device__ __noinline__
void SigMismatch_Handler() {
    printf("Errore ASPIS: Signature mismatch rilevata\n");
    asm("trap;");
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//sum function to test handling of device functions by eddi pass.
//nvcc would inline such function and resolve any argument duplication and
//rettoref
__device__
float sum(float x, float y)
{
    return x+y;
}
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = sum(a*x[i] ,y[i]); 
}

int main(void){
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;

    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 1000) / 10);
    printf("CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 1000) / 10);

    x = (float*) malloc(N*sizeof(float));
    y = (float*) malloc(N*sizeof(float));

    gpuErrchk(cudaMalloc(&d_x, N*sizeof(float)));
    gpuErrchk(cudaMalloc(&d_y, N*sizeof(float)));

    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    gpuErrchk(cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice));

    saxpy<<<(N+255)/256, 256>>> (N, 2.0f, d_x, d_y);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for (int i=0; i<N; i++){
        // printf("y[%d] = %f\n", i, y[i]);
        maxError = std::max(maxError, std::abs(y[i]-4.0f));
    }
        printf("Max error: %f\n", maxError);
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

}
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include "classify_cuda.cuh"

/*
 * Arguments:
 * data: Memory that contains both the review LSA coefficients and the labels.
 *       Format decided by implementation of classify.
 * batch_size: Size of mini-batch, how many elements to process at once
 * step_size: Step size for gradient descent. Tune this as needed. 1.0 is sane
 *            default.
 * weights: Pointer to weights vector of length REVIEW_DIM.
 * errors: Pointer to a single float used to describe the error for the batch.
 *         An output variable for the kernel. The kernel can either write the
 *         value of loss function over the batch or the misclassification rate
 *         in the batch to errors.
 */
__device__ static float atomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
	float *weights,
    float *errors)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    // if (threadIdx.x == 0) printf("thread_index: %d\n", thread_index);
    // __shared__ float gradient[50];
    __shared__ float gradient[1024];
    // __shared__ float er;
    while (thread_index < batch_size) {
        float wx = 0.0;
        for (int i = 0; i < REVIEW_DIM; ++i) {
            wx += weights[i] * data[thread_index*(REVIEW_DIM+1)+i];
        }
        float denom = (1 + exp(data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * wx));
        gradient[threadIdx.x] = (-1.0/batch_size * data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] * data[thread_index*(REVIEW_DIM+1)+i])/denom;
        
        // printf("%f\n", wx);
        
        // if (wx * data[thread_index*(REVIEW_DIM+1)+REVIEW_DIM] < 0) {
        //     atomicAdd(&er, 1.0);
        //     // printf("%f\n", er);
        // }
        // printf("wx: %f\n", wx);
        
        // float temp[50];
        // if (threadIdx.x == 0) {
        //     for (int i = 0; i < REVIEW_DIM; ++i) {
        //         weights[i] -= step_size * gradient[i];
        //     // printf("%f\n", weights[i]);
        //     }
        // }
        // printf("gra: ", gradient[0]);
        // printf("\n");
        *errors = 2.0;
        return;
        thread_index += gridDim.x * blockDim.x;
    }
    if (threadIdx.x == 0) {
        
        // *errors = er / batch_size;
        for (int i = 0; i < REVIEW_DIM; ++i) {
            weights[i] -= step_size * gradient[i];
        // printf("%f\n", weights[i]);
        }
    }
   
}

/*
 * All parameters have the same meaning as in docstring for trainLogRegKernel.
 * Notably, cudaClassify returns a float that quantifies the error in the
 * minibatch. This error should go down as more training occurs.
 */
float cudaClassify(
    float *data, // dev_data[0,1]
    int batch_size, // batch_size = 2048
    float step_size, // step_size = 1.0
    float *weights,  // dev_weights
    cudaStream_t stream) // s[0,1]
{
    int block_size = (batch_size < 1024) ? batch_size : 1024;

    // grid_size = CEIL(batch_size / block_size)
    int grid_size = (batch_size + block_size - 1) / block_size;
    int shmem_bytes = 0;

    float *d_errors;
    cudaMalloc(&d_errors, sizeof(float));
    cudaMemset(d_errors, 0, sizeof(float));
    printf("entering kernel\n");
    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);
    printf("leaving kernel\n");
    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}

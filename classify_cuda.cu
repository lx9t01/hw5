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
__global__
void trainLogRegKernel(
    float *data,
    int batch_size,
    int step_size,
	float *weights,
    float *errors)
{
    unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("thread_index: %d\n", thread_index);
    float wx[2048];
    float denom[2048];
    float losslog[2048];
    float temp[2048 * 50];

    while (thread_index < batch_size) {
        wx[thread_index] = 1.0;
        
        __syncthreads();
        for (int i = 0; i < REVIEW_DIM; ++i) {
            wx[thread_index] += weights[i] * data[thread_index * REVIEW_DIM + i];

        }
        __syncthreads();
        denom[thread_index] = 1.0 + exp (data[thread_index * REVIEW_DIM + REVIEW_DIM] * wx[thread_index]);
        losslog[thread_index] = log(1.0 + exp (-data[thread_index * REVIEW_DIM + REVIEW_DIM] * wx[thread_index])) / batch_size;
        __syncthreads();
        for (int i = 0; i < REVIEW_DIM; ++i) {
            temp[thread_index * REVIEW_DIM + i] = -data[thread_index * REVIEW_DIM + REVIEW_DIM] * data[thread_index * REVIEW_DIM + i] / denom[thread_index] / batch_size;
        }
        __syncthreads();
        thread_index += gridDim.x * blockDim.x;
    }

    int l = batch_size;
    while (l > 1) {
        l /= 2;
        for (int j = 0; j < REVIEW_DIM; ++j) {
            if (thread_index < l) {
                temp[thread_index * REVIEW_DIM + j] = \
                    temp[thread_index * REVIEW_DIM + j] + temp[(thread_index + l) * REVIEW_DIM + j];
            }
        }
        if (thread_index < l) {
            losslog[thread_index] += losslog[thread_index + l];
        }
        __syncthreads();
    }

    for (int i = 0; i < REVIEW_DIM; ++i) {
        weights[i] = weights[i] - step_size * temp[i];
    }
    *errors = 1.0;    
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

    trainLogRegKernel<<<grid_size, block_size, shmem_bytes, stream>>>(
        data,
        batch_size,
        step_size,
        weights,
        d_errors);

    float h_errors = -1.0;
    cudaMemcpy(&h_errors, d_errors, sizeof(float), cudaMemcpyDefault);
    cudaFree(d_errors);
    return h_errors;
}

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

#include "classify_cuda.cuh"
#include "ta_utilities.hpp"

using namespace std;

/*
NOTE: You can use this macro to easily check cuda error codes
and get more information.

Modified from:
http://stackoverflow.com/questions/14038589/
what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
*/
#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(
    cudaError_t code,
    const char *file,
    int line,
    bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n",
            cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// timing setup code
cudaEvent_t start;
cudaEvent_t stop;

#define START_TIMER() {                         \
    gpuErrChk(cudaEventCreate(&start));         \
    gpuErrChk(cudaEventCreate(&stop));          \
    gpuErrChk(cudaEventRecord(start));          \
}

#define STOP_RECORD_TIMER(name) {                           \
    gpuErrChk(cudaEventRecord(stop));                       \
    gpuErrChk(cudaEventSynchronize(stop));                  \
    gpuErrChk(cudaEventElapsedTime(&name, start, stop));    \
    gpuErrChk(cudaEventDestroy(start));                     \
    gpuErrChk(cudaEventDestroy(stop));                      \
}

////////////////////////////////////////////////////////////////////////////////
// Start non boilerplate code

// Fills output with standard normal data
void gaussianFill(float *output, int size) { // size = 2048
    // seed generator to 2015
    std::default_random_engine generator(2015);
    std::normal_distribution<float> distribution(0.0, 0.1);
    for (int i=0; i < size; i++) {
        output[i] = distribution(generator);
    }
}

// Takes a string of comma seperated floats and stores the float values into
// output. Each string should consist of REVIEW_DIM + 1 floats.
void readLSAReview(string review_str, float *output, int stride) { 
    stringstream stream(review_str);
    int component_idx = 0;

    for (string component; getline(stream, component, ','); component_idx++) {
        output[stride * component_idx] = atof(component.c_str());
    }
    assert(component_idx == REVIEW_DIM + 1); 
}

void classify(istream& in_stream, int batch_size) {
    // TODO: randomly initialize weights. allocate and initialize buffers on
    //       host & device. allocate and initialize streams
    float* weights = (float*) malloc (REVIEW_DIM * sizeof(float));
    gaussianFill (weights, REVIEW_DIM);
    // float* dev_weights;
    // cudaMalloc(&dev_weights, REVIEW_DIM * sizeof(float));


    // number of streams = 2
    const int num_streams = 2;
    // buffers & streams
    float* dev_data[num_streams];
    float* host_data[num_streams];
    float* host_error[num_streams];
    cudaStream_t s[num_streams];
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&s[i]);
        cudaMalloc(&dev_data[i], batch_size * (REVIEW_DIM + 1) * sizeof(float));
        dev_data[i] = (float*)malloc(batch_size * (REVIEW_DIM + 1) * sizeof(float));
        host_error[i] = (float*)malloc(sizeof(float));
    }

    // main loop to process input lines (each line corresponds to a review)
    int review_idx = 0;
    for (string review_str; getline(in_stream, review_str); review_idx++) {
        // TODO: process review_str with readLSAReview
        for (int i = 0; i < num_streams; ++i) {
            readLSAReview(review_str, host_data[i][review_idx], 1)； // what is the stride here = 1
            // TODO: if you have filled up a batch, copy H->D, call kernel and copy
            if (review_idx >= batch_size - 1) {
            // copy from host to device
                cudaMemcpyAsync(dev_data[i], host_data[i], 
                    batch_size * (REVIEW_DIM + 1) * sizeof(float), cudaMemcpyHostToDevice, s[i]);
                host_error[i] = cudaClassify(dev_data, batch_size, 1.0, weights, s[i]);
                review_idx = 0;
                //      D->H all in a stream
                printf("error rate: %f\n", host_error[i]);
            }
        }
        
    }

    // TODO: print out weights
    for (int i = 0; i < REVIEW_DIM; ++i) {
        printf("%f ", weight[i]);
    }
    printf("\n");
    // TODO: free all memory

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
		printf("./classify <path to datafile>\n");
		return -1;
    } 
    // These functions allow you to select the least utilized GPU
    // on your system as well as enforce a time limit on program execution.
    // Please leave these enabled as a courtesy to your fellow classmates
    // if you are using a shared computer. You may ignore or remove these
    // functions if you are running on your local machine.
    TA_Utilities::select_least_utilized_GPU();
    int max_time_allowed_in_seconds = 100;
    TA_Utilities::enforce_time_limit(max_time_allowed_in_seconds);
	
    // Init timing
	float time_initial, time_final;
	
    int batch_size = 2048;
	
	// begin timer
	time_initial = clock();
	
    ifstream ifs(argv[1]);
    stringstream buffer;
    buffer << ifs.rdbuf();
    classify(buffer, batch_size);
	
	// End timer
	time_final = clock();
	printf("Total time to run classify: %f (s)\n", (time_final - time_initial) / CLOCKS_PER_SEC);
	

}
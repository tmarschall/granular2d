/*
 * error_check.cpp
 *
 *  Created on: Apr 10, 2014
 *      Author: marschat
 */


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdio>


void check_last_error(const char *msg = "")
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

void check_error(cudaError_t err, const char *msg = "")
{
	cudaDeviceSynchronize();
	if( cudaSuccess != err) {
			fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
			exit(EXIT_FAILURE);
	}
}



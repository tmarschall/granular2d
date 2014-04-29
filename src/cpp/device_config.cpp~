/*
 * device_config.cpp
 *
 *  Created on: Apr 10, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include "device_config.h"
#include "error_check.h"
#include <cassert>
#include <cstdio>


cudaDeviceProp get_device_info()
{
	int d = 0;
	check_error(cudaGetDevice(&d));
	struct cudaDeviceProp props;
	check_error(cudaGetDeviceProperties(&props, d));
	return props;
}

void print_device_info()
{
	cudaDeviceProp props = get_device_info();
	printf("Using Device: %s\n", props.name);
	printf("Number of multiprocessors: %d, Warp size: %d, Max threads per block: %d\n",
			props.multiProcessorCount, props.warpSize, props.maxThreadsPerBlock);
	printf("Global Mem: %g MB, Shared Mem per block: %g KB, Registers per block: %d\n",
			props.totalGlobalMem/(1024.*1024.), props.sharedMemPerBlock/(1024.), props.regsPerBlock);
}

/* Configure kernel for current device
 * Arguements: np - number of particles
 * 			   tpp - number of threads per particle
 *			   smpt - shared mem (bytes) per thread
 *			   smpb - fixed shared mem per block (doesn't scale with threads)
 */
void KernelConfig::configure(int np, int tpp, int smpt, int smpb)
{
	cudaDeviceProp props = get_device_info();
	assert(smpb < props.sharedMemPerBlock);
	int tpb = props.warpSize*tpp;
	// Generally good to have somewhere around 128 to 256 threads per block
	while (tpb <= 128) {
		tpb *= 2;
	}
	while (tpb > 256) {
		tpb /= 2;
	}
	int nb = (np*tpp + tpb - 1) / tpb;
	int sm = smpb + tpb * smpt;
	// Want to have more than 2 blocks per multiprocessor (even more usually better)
	// And can't exceed the device limit of shared memory per block
	while (nb < 3*props.multiProcessorCount || sm > props.sharedMemPerBlock) {
		tpb /= 2;
		nb *= 2;
		sm = smpb + tpb * smpt;
	}

	// Extreme cases
	if (tpb < props.warpSize) {
		fprintf(stderr, "Trying to allocate too much shared memory per block, check parameters");
		exit(1);
	}
	if (nb > props.maxGridSize[2]) {
		fprintf(stderr, "Trying to allocate too many blocks, you may need to reduce particle number");
		exit(1);
	}

	m_grid_dim = dim3(nb);
	m_block_dim = dim3(tpb);
	m_shared_mem = sm;
}



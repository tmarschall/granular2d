/*
 * device_config.h
 *
 *  Created on: Apr 10, 2014
 *      Author: marschat
 */

#ifndef DEVICE_CONFIG_H_
#define DEVICE_CONFIG_H_

#include <cuda.h>
#include <cuda_runtime.h>

cudaDeviceProp get_device_info();
// Print info on the currently selected cuda-capable gpu
void print_device_info();

class KernelConfig
{
private:
	dim3 m_grid_dim;
	dim3 m_block_dim;
	size_t m_shared_mem;

public:
	KernelConfig(void) { m_grid_dim = dim3(1); m_block_dim = dim3(32); m_shared_mem = 0; }
	KernelConfig(int g, int b, int s = 0) { m_grid_dim = dim3(g); m_block_dim = dim3(b); m_shared_mem = s; }
	KernelConfig(dim3 g = dim3(), dim3 b = dim3(), int s = 0) { m_grid_dim = g; m_block_dim = b; m_shared_mem = s; }

	void configure(int np, int tpp, int smpt = 0, int smpb = 0);

	inline dim3 grid_dim() { return m_grid_dim; }
	inline dim3 block_dim() { return m_block_dim; }
	inline size_t shared_mem() { return m_shared_mem; }
	void grid_dim(dim3 g) { m_grid_dim = g; }
	void block_dim(dim3 b) { m_block_dim = b; }
	void shared_mem(size_t s) { m_shared_mem = s; }
};


#endif /* DEVICE_CONFIG_H_ */

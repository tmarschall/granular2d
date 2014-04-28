/*
 * disk.cpp
 *
 *  Created on: Feb 25, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "disk.h"

/*
 * Disk Array Functions
 */

DiskArray::DiskArray(int nDisks)
{
	m_nDisks = nDisks;
	cudaHostAlloc((void**) &h_pd2Pos, m_nDisks*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nDisks*sizeof(double2));
	cudaHostAlloc((void**) &h_pdR, m_nDisks*sizeof(double), 0);
	cudaMalloc((void**) &d_pdR, m_nDisks*sizeof(double));
}

DiskArray::DiskArray(int nDisks, double2 *pd2Pos, double *pdR)
{
	m_nDisks = nDisks;
	cudaHostAlloc((void**) &h_pd2Pos, m_nDisks*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nDisks*sizeof(double2));
	cudaHostAlloc((void**) &h_pdR, m_nDisks*sizeof(double), 0);
	cudaMalloc((void**) &d_pdR, m_nDisks*sizeof(double));

	for (int d = 0; d < m_nDisks; d++) {
		h_pd2Pos[d] = pd2Pos[d];
		h_pdR[d] = pdR[d];
	}
	cpyAllToDevice();
}

DiskArray::~DiskArray() {
	cudaFreeHost(h_pd2Pos);
	cudaFree(d_pd2Pos);
	cudaFreeHost(h_pdR);
	cudaFree(d_pdR);
}

void DiskArray::cpyAllToDevice() {
	cudaMemcpyAsync(d_pdR, h_pdR, m_nDisks*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

void DiskArray::cpyAllToHost() {
	cudaMemcpyAsync(h_pdR, d_pdR, m_nDisks*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
}

void DiskArray::cpyPosToDevice() {
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyHostToDevice);
	cudaThreadSynchronize();
}

void DiskArray::cpyPosToHost() {
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
}

double DiskArray::dArea() {
	double dA = 0.0;
	for (int d = 0; d < m_nDisks; d++) {
		dA += h_pdR[d]*h_pdR[d];
	}
	dA *= 0.5*D_PI;

	return dA;
}





IdenticalDiskArray::IdenticalDiskArray(int nDisks)
{
	m_nDisks = nDisks;
	cudaHostAlloc((void**) &h_pd2Pos, m_nDisks*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nDisks*sizeof(double2));
	m_dR = 0.5;
}

IdenticalDiskArray::IdenticalDiskArray(int nDisks, double2 *pd2Pos, double dR)
{
	m_nDisks = nDisks;
	cudaHostAlloc((void**) &h_pd2Pos, m_nDisks*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nDisks*sizeof(double2));
	m_dR = dR;

	for (int d = 0; d < m_nDisks; d++) {
		h_pd2Pos[d] = pd2Pos[d];
	}
	cpyAllToDevice();
}

IdenticalDiskArray::~IdenticalDiskArray() {
	cudaFreeHost(h_pd2Pos);
	cudaFree(d_pd2Pos);
}

void IdenticalDiskArray::cpyAllToDevice() {
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void IdenticalDiskArray::cpyAllToHost() {
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
}

void IdenticalDiskArray::cpyPosToDevice() {
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void IdenticalDiskArray::cpyPosToHost() {
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nDisks*sizeof(double2), cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
}


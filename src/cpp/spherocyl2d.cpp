/*
 * spherocyl2d.cpp
 *
 *  Created on: Feb 25, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include "spherocyl2d.h"


Spherocyl2DArray::Spherocyl2DArray(int nSpherocyls)
{
	m_nSpherocyls = nSpherocyls;
	cudaHostAlloc((void**) &h_pd2Pos, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nSpherocyls*sizeof(double2));
	cudaHostAlloc((void**) &h_pdPhi, m_nSpherocyls*sizeof(double), 0);
	cudaMalloc((void**) &d_pdPhi, m_nSpherocyls*sizeof(double));
	cudaHostAlloc((void**) &h_pd2Geo, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Geo, m_nSpherocyls*sizeof(double2));
}


Spherocyl2DArray::Spherocyl2DArray(int nSpherocyls, double2 *pd2Pos, double *pdPhi, double2 *pd2Geo)
{
	m_nSpherocyls = nSpherocyls;
	cudaHostAlloc((void**) &h_pd2Pos, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nSpherocyls*sizeof(double2));
	cudaHostAlloc((void**) &h_pdPhi, m_nSpherocyls*sizeof(double), 0);
	cudaMalloc((void**) &d_pdPhi, m_nSpherocyls*sizeof(double));
	cudaHostAlloc((void**) &h_pd2Geo, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Geo, m_nSpherocyls*sizeof(double2));

	for (int d = 0; d < m_nSpherocyls; d++) {
			h_pd2Pos[d] = pd2Pos[d];
			h_pdPhi[d] = pdPhi[d];
			h_pd2Geo[d] = pd2Geo[d];
		}
	cpyAllToDevice();
}

Spherocyl2DArray::~Spherocyl2DArray()
{
	cudaFreeHost(h_pd2Pos);
	cudaFree(d_pd2Pos);
	cudaFreeHost(h_pdPhi);
	cudaFree(d_pdPhi);
	cudaFreeHost(h_pd2Geo);
	cudaFree(d_pd2Geo);
}

void Spherocyl2DArray::cpyAllToDevice() {
	cudaMemcpyAsync(d_pdPhi, h_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Geo, h_pd2Geo, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void Spherocyl2DArray::cpyAllToHost() {
	cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_pd2Geo, d_pd2Geo, m_nSpherocyls*sizeof(double2), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
}

void Spherocyl2DArray::cpyPosToDevice() {
	cudaMemcpyAsync(d_pdPhi, h_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void Spherocyl2DArray::cpyPosToHost() {
	cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToHost);
	//cudaThreadSynchronize();
}

double Spherocyl2DArray::dArea() {
	double dA = 0.0;
	for (int d = 0; d < m_nSpherocyls; d++) {
		dA += h_pd2Geo[d].x*(4*h_pd2Geo[d].y + 0.5*D_PI*h_pd2Geo[d].x);
	}

	return dA;
}



IdenticalSpherocyl2DArray::IdenticalSpherocyl2DArray(int nSpherocyls)
{
	m_nSpherocyls = nSpherocyls;
	cudaHostAlloc((void**) &h_pd2Pos, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nSpherocyls*sizeof(double2));
	cudaHostAlloc((void**) &h_pdPhi, m_nSpherocyls*sizeof(double), 0);
	cudaMalloc((void**) &d_pdPhi, m_nSpherocyls*sizeof(double));
	m_dR = 0.5;
	m_dA = 2.0;
}


IdenticalSpherocyl2DArray::IdenticalSpherocyl2DArray(int nSpherocyls, double2 *pd2Pos, double *pdPhi, double dR, double dA)
{
	m_nSpherocyls = nSpherocyls;
	cudaHostAlloc((void**) &h_pd2Pos, m_nSpherocyls*sizeof(double2), 0);
	cudaMalloc((void**) &d_pd2Pos, m_nSpherocyls*sizeof(double2));
	cudaHostAlloc((void**) &h_pdPhi, m_nSpherocyls*sizeof(double), 0);
	cudaMalloc((void**) &d_pdPhi, m_nSpherocyls*sizeof(double));
	m_dR = dR;
	m_dA = dA;

	for (int d = 0; d < m_nSpherocyls; d++) {
			h_pd2Pos[d] = pd2Pos[d];
			h_pdPhi[d] = pdPhi[d];
		}
	cpyAllToDevice();
}

IdenticalSpherocyl2DArray::~IdenticalSpherocyl2DArray()
{
	cudaFreeHost(h_pd2Pos);
	cudaFree(d_pd2Pos);
	cudaFreeHost(h_pdPhi);
	cudaFree(d_pdPhi);
}

void IdenticalSpherocyl2DArray::cpyAllToDevice() {
	cudaMemcpyAsync(d_pdPhi, h_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void IdenticalSpherocyl2DArray::cpyAllToHost() {
	cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyDeviceToHost);
	//cudaThreadSynchronize();
}

void IdenticalSpherocyl2DArray::cpyPosToDevice() {
	cudaMemcpyAsync(d_pdPhi, h_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_pd2Pos, h_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToDevice);
	//cudaThreadSynchronize();
}

void IdenticalSpherocyl2DArray::cpyPosToHost() {
	cudaMemcpyAsync(h_pdPhi, d_pdPhi, m_nSpherocyls*sizeof(double), cudaMemcpyHostToHost);
	cudaMemcpyAsync(h_pd2Pos, d_pd2Pos, m_nSpherocyls*sizeof(double2), cudaMemcpyHostToHost);
	//cudaThreadSynchronize();
}

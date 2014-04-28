/*
 * neighbor_list.cpp
 *
 *  Created on: Mar 17, 2014
 *      Author: marschat
 */

#include "neighbor_list.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

NeighborList::NeighborList(int nParticles, int nMaxNbrs)
{
	m_nParticles = nParticles;
	m_nMaxNbrs = nMaxNbrs;
	h_pnNbrList = new int[m_nParticles*m_nMaxNbrs];
	cudaMalloc((void**)&d_pnNbrList, m_nParticles*m_nMaxNbrs*sizeof(int));
}

NeighborList::~NeighborList()
{
	delete[] h_pnNbrList;
	cudaFree(d_pnNbrList);
}





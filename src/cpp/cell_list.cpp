/*
 * cell_list.cpp
 *
 *  Created on: Mar 11, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <vector_types.h>
#include "cell_list.h"

CellList2D::CellList2D(int nCells, int nParticles, int nMaxPPC)
{
	m_nCells = nCells;
	m_n2GridDim = make_int2(m_nCells, 1);
	m_nParticles = nParticles;
	m_nMaxPPC = nMaxPPC;
	h_pnParticleCells = new int[nParticles];
	h_pnCellCounts = new int[nCells];
	h_pnCellLists = new int[nCells*nMaxPPC];
	cudaMalloc((void**)&d_pnParticleCells, nParticles*sizeof(int));
	cudaMalloc((void**)&d_pnCellCounts, nCells*sizeof(int));
	cudaMalloc((void**)&d_pnCellLists, nCells*nMaxPPC*sizeof(int));
}

CellList2D::CellList2D(int2 n2GridDim, int nParticles, int nMaxPPC)
{
	m_n2GridDim = n2GridDim;
	m_nCells = n2GridDim.x*n2GridDim.y;
	m_nParticles = nParticles;
	m_nMaxPPC = nMaxPPC;
	h_pnParticleCells = new int[m_nParticles];
	h_pnCellCounts = new int[m_nCells];
	h_pnCellLists = new int[m_nCells*m_nMaxPPC];
	cudaMalloc((void**)&d_pnParticleCells, m_nParticles*sizeof(int));
	cudaMalloc((void**)&d_pnCellCounts, m_nCells*sizeof(int));
	cudaMalloc((void**)&d_pnCellLists, m_nCells*nMaxPPC*sizeof(int));
}

CellList2D::~CellList2D()
{
	delete[] h_pnParticleCells;
	delete[] h_pnCellCounts;
	delete[] h_pnCellLists;
	cudaFree(d_pnParticleCells);
	cudaFree(d_pnCellCounts);
	cudaFree(d_pnCellLists);
}

void CellList2D::cpyListToDevice()
{
	cudaMemcpy(d_pnParticleCells, h_pnParticleCells, m_nParticles*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pnCellCounts, h_pnCellCounts, m_nCells*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_pnCellLists, h_pnCellLists, m_nCells*m_nMaxPPC*sizeof(int), cudaMemcpyHostToDevice);
}

void CellList2D::cpyListToHost()
{
	cudaMemcpy(h_pnParticleCells, d_pnParticleCells, m_nParticles*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pnCellCounts, d_pnCellCounts, m_nCells*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_pnCellLists, d_pnCellLists, m_nCells*m_nMaxPPC*sizeof(int), cudaMemcpyDeviceToHost);
}

void CellList2D::setCells(int nCells)
{
	delete[] h_pnCellCounts;
	delete[] h_pnCellLists;
	cudaFree(d_pnCellCounts);
	cudaFree(d_pnCellLists);
	m_nCells = nCells;
	m_n2GridDim = make_int2(nCells, 1);
	h_pnCellCounts = new int[nCells];
	h_pnCellLists = new int[nCells*m_nMaxPPC];
}
void CellList2D::setGridDim(int2 n2GridDim)
{
	delete[] h_pnCellCounts;
	delete[] h_pnCellLists;
	cudaFree(d_pnCellCounts);
	cudaFree(d_pnCellLists);
	m_n2GridDim = n2GridDim;
	m_nCells = n2GridDim.x*n2GridDim.y;

}
void CellList2D::setGridDimX(int nDim)
{
	delete[] h_pnCellCounts;
	delete[] h_pnCellLists;
	cudaFree(d_pnCellCounts);
	cudaFree(d_pnCellLists);
}
void CellList2D::setGridDimY(int nDim)
{
	delete[] h_pnCellCounts;
	delete[] h_pnCellLists;
	cudaFree(d_pnCellCounts);
	cudaFree(d_pnCellLists);
}
void CellList2D::setParticles(int nParticles)
{
	delete[] h_pnParticleCells;
	cudaFree(d_pnParticleCells);

}
void CellList2D::setMaxPPC(int nMaxPPC)
{
	delete[] h_pnCellLists;
	cudaFree(d_pnCellLists);
}


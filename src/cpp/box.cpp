/*
 * box.cpp
 *
 *  Created on: Feb 26, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "box.h"
#include <math.h>
#include <algorithm>


void Box2D::configure_cells()
{
	double dWMin = 1.12 * m_dMinCellD;
	double dHMin = m_dMinCellD;

	m_nRows = std::max(static_cast<int>(m_dLy / dHMin), 3);
	m_nCols = std::max(static_cast<int>(m_dLx/ dWMin), 3);
	m_nCells = m_nRows * m_nCols;

	m_dCellW = m_dLx / m_nCols;
	m_dCellH = m_dLy / m_nRows;

	h_pnCellNbrs = new int[8*m_nCells];
	cudaMalloc((void **) &d_pnCellNbrs, 8*m_nCells*sizeof(int));
	for (int k = 0; k < m_nCells; k++) {
		int r = k / m_nCols;
		int c = k % m_nCols;

		int nAdjCol1 = (c + 1) % m_nCols;
		int nAdjCol2 = (m_nCols + c - 1) % m_nCols;
		h_pnCellNbrs[8 * k] = r * m_nCols + nAdjCol1;
		h_pnCellNbrs[8 * k + 1] = r * m_nCols + nAdjCol2;

		int nAdjRow = (r + 1) % m_nRows;
		h_pnCellNbrs[8 * k + 2] = nAdjRow * m_nCols + c;
		h_pnCellNbrs[8 * k + 3] = nAdjRow * m_nCols + nAdjCol1;
		h_pnCellNbrs[8 * k + 4] = nAdjRow * m_nCols + nAdjCol2;

		nAdjRow = (m_nRows + r - 1) % m_nRows;
		h_pnCellNbrs[8 * k + 5] = nAdjRow * m_nCols + c;
		h_pnCellNbrs[8 * k + 6] = nAdjRow * m_nCols + nAdjCol1;
		h_pnCellNbrs[8 * k + 7] = nAdjRow * m_nCols + nAdjCol2;
	}
    cudaMemcpy(d_pnCellNbrs, h_pnCellNbrs, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
}

Box2D::Box2D(double dLx, double dLy, double dGamma, double dMinCellD)
{
	m_dLx = dLx;
	m_dLy = dLy;
	m_dGamma = dGamma;
	if (dMinCellD == 0) {
		m_dMinCellD = fmax(dLx, dLy);
	}

	configure_cells();
	cpyCellsToDevice();
}

Box2D::~Box2D()
{
    delete[] h_pnCellNbrs;
    cudaFree(d_pnCellNbrs);
}

void Box2D::cpyCellsToDevice()
{
	cudaMemcpy(d_pnCellNbrs, h_pnCellNbrs, 8*m_nCells*sizeof(int), cudaMemcpyHostToDevice);
}

void Box2D::cpyCellsToHost()
{
	cudaMemcpy(h_pnCellNbrs, d_pnCellNbrs, 8*m_nCells*sizeof(int), cudaMemcpyDeviceToHost);
}

void Box2D::setLx(double dLx)
{
    m_dLx = dLx;
    double dWMin = 1.12 * m_dMinCellD;
	int nCols = std::max(static_cast<int>(m_dLx / dWMin), 3);
    if (nCols != m_nCols) {
        delete[] h_pnCellNbrs;
        cudaFree(d_pnCellNbrs);
        configure_cells();
    }
    
}

void Box2D::setLy(double dLy)
{
    m_dLy = dLy;
    double dHMin = m_dMinCellD;
	int nRows = std::max(static_cast<int>(m_dLy / dHMin), 3);
    if (nRows != m_nRows) {
        delete[] h_pnCellNbrs;
        cudaFree(d_pnCellNbrs);
        configure_cells();
    }
}

void Box2D::setMinCellD(double dMinCellD)
{
    m_dMinCellD = dMinCellD;
    
    double dWMin = 1.12 * m_dMinCellD;
	double dHMin = m_dMinCellD;
	int nRows = std::max(static_cast<int>(m_dLy / dHMin), 3);
	int nCols = std::max(static_cast<int>(m_dLx/ dWMin), 3);
    if (nRows != m_nRows || nCols != m_nCols) {
        delete[] h_pnCellNbrs;
        cudaFree(d_pnCellNbrs);
        configure_cells();
    }
}

void Box2D::rescaleL(double dScaler) {
    m_dLx *= dScaler;
    m_dLy *= dScaler;
	int nRows = std::max(static_cast<int>(m_dLy / m_dMinCellD), 3);
	int nCols = std::max(static_cast<int>(m_dLx/ (1.12*m_dMinCellD)), 3);
    if (nRows != m_nRows || nCols != m_nCols) {
        delete[] h_pnCellNbrs;
        cudaFree(d_pnCellNbrs);
        configure_cells();
    }
    
}





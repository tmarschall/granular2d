/*
 * find_neighbors.cu
 *
 *  Created on: Apr 22, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include "find_neighbors.h"
#include "error_check.h"


__global__ void find_cells(int nParticles, int nMaxPPC, double dCellW, double dCellH,
						   int nCellCols, double dLx, double dLy, double2 *pd2Pos,
						   int *pnCellID, int *pnPPC, int *pnCellList)
{
  // Assign each thread a unique ID accross all thread-blocks, this is its particle ID
  int nPID = threadIdx.x + blockIdx.x * blockDim.x;
  int nThreads = blockDim.x * gridDim.x;

  while (nPID < nParticles) {
    double dX = pd2Pos[nPID].x;
    double dY = pd2Pos[nPID].y;

    // I often allow the stored coordinates to drift slightly outside the box limits
    //  until
    if (dY > dLy)
      {
    	dY -= dLy;
    	pd2Pos[nPID].y = dY;
      }
    else if (dY < 0)
      {
    	dY += dLy;
    	pd2Pos[nPID].y = dY;
      }
    if (dX > dLx)
      {
    	dX -= dLx;
    	pd2Pos[nPID].x = dX;
      }
    else if (dX < 0)
      {
    	dX += dLx;
		pd2Pos[nPID].x = dX;
      }

    //find the cell ID, add a particle to that cell
    int nCol = (int)(dX / dCellW);
    int nRow = (int)(dY / dCellH);
    int nCellID = nCol + nRow * nCellCols;
    pnCellID[nPID] = nCellID;

    // Add 1 particle to a cell safely (only allows one thread to access the memory
    //  address at a time). nPPC is the original value, not the result of addition
    int nPPC = atomicAdd(pnPPC + nCellID, 1);

    // only add particle to cell if there is not already the maximum number in cell
    if (nPPC < nMaxPPC)
      pnCellList[nCellID * nMaxPPC + nPPC] = nPID;
    else
      nPPC = atomicAdd(pnPPC + nCellID, -1);

    nPID += nThreads;
  }
}

void find_cells(CellList2D &cl, Box2D &b, DiskArray &d, KernelConfig k)
{
	find_cells <<<k.grid_dim(), k.block_dim()>>>
			(d.nDisks(), cl.nMaxPPC(), b.dCellW(), b.dCellH(), b.nCols(), b.dLx(), b.dLy(), d.devPos(),
			 cl.devParticleCells(), cl.devCellCounts(), cl.devCellLists());
	check_last_error("Finding disks' cells");
}

void find_cells(CellList2D &cl, Box2D &b, IdenticalDiskArray &d, KernelConfig k)
{
	find_cells <<<k.grid_dim(), k.block_dim()>>>
			(d.nDisks(), cl.nMaxPPC(), b.dCellW(), b.dCellH(), b.nCols(), b.dLx(), b.dLy(), d.devPos(),
			 cl.devParticleCells(), cl.devCellCounts(), cl.devCellLists());
	check_last_error("Finding disks' cells");
}

void find_cells(CellList2D &cl, Box2D &b, Spherocyl2DArray &s, KernelConfig k)
{
	find_cells <<<k.grid_dim(), k.block_dim()>>>
			(s.nSpherocyls(), cl.nMaxPPC(), b.dCellW(), b.dCellH(), b.nCols(), b.dLx(), b.dLy(), s.devPos(),
			 cl.devParticleCells(), cl.devCellCounts(), cl.devCellLists());
	check_last_error("Finding spherocyls' cells");
}

void find_cells(CellList2D &cl, Box2D &b, IdenticalSpherocyl2DArray &s, KernelConfig k)
{
	find_cells <<<k.grid_dim(), k.block_dim()>>>
			(s.nSpherocyls(), cl.nMaxPPC(), b.dCellW(), b.dCellH(), b.nCols(), b.dLx(), b.dLy(), s.devPos(),
			 cl.devParticleCells(), cl.devCellCounts(), cl.devCellLists());
	check_last_error("Finding spherocyls' cells");
}

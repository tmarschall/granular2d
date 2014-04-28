/*
 * cell_list.h
 *
 *  Created on: Mar 11, 2014
 *      Author: marschat
 */

#ifndef CELL_LIST_H_
#define CELL_LIST_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

class CellList2D
{
private:
	int m_nCells;
	int m_nParticles;
	int m_nMaxPPC;
	int2 m_n2GridDim;
	int *h_pnParticleCells;
	int *d_pnParticleCells;
	int *h_pnCellCounts;
	int *d_pnCellCounts;
	int *h_pnCellLists;
	int *d_pnCellLists;
public:
	CellList2D(int nCells, int nParticles, int nMaxPPC);
	CellList2D(int2 n2GridDim, int nParticles, int nMaxPPC);
	~CellList2D();

	void cpyListToDevice();
	void cpyListToHost();

	inline int nCells() { return m_nCells; }
	inline int nParticles() { return m_nParticles; }
	inline int nMaxPPC() { return m_nMaxPPC; }
	inline int2 n2GridDim() { return m_n2GridDim; }
	inline int nRows() { return m_n2GridDim.x; }
	inline int nCols() { return m_n2GridDim.y; }
	inline int nCell(int idx) { return h_pnParticleCells[idx]; }
	int *devParticleCells() { return d_pnParticleCells; }
	int *devCellCounts() { return d_pnCellCounts; }
	int *devCellLists() { return d_pnCellLists; }

	void setCells(int nCells);
	void setGridDim(int2 n2GridDim);
	void setGridDimX(int nDim);
	void setGridDimY(int nDim);
	void setParticles(int nParticles);
	void setMaxPPC(int nMaxPPC);
};


/*  TODO:
 *
 * 	Rather than having a max # of particles per cell and fixed amount of memory for each cell list
 * 	store the cell assignments more compactly with the particle indices reordered by cell
 * 	and the start index of each cell stored in another list
 *
 * 	Might be necessary/more efficient for larger system sizes
 *
 */
class CompactCellList2D
{

};


#endif /* CELL_LIST_H_ */

/*
 * box.h
 *
 *  Created on: Feb 25, 2014
 *      Author: marschat
 */

#ifndef BOX_H_
#define BOX_H_


class Box2D
{
private:
	double m_dLx;
	double m_dLy;
	double m_dGamma;

	int m_nCells;
	int m_nRows;
	int m_nCols;
	double m_dMinCellD;  // Minimum distance across cell such that particles may only contact adjacent cells
	double m_dCellW;
	double m_dCellH;

	int *h_pnCellNbrs;
	int *d_pnCellNbrs;

	void configure_cells();

public:
	Box2D(double dLx, double dLy, double dGamma, double dMinCellD = 0);
	~Box2D();

	void cpyCellsToDevice();
	void cpyCellsToHost();

	// Getters
	inline double dLx() { return m_dLx; }
	inline double dLy() { return m_dLy; }
	inline double dCellW() { return m_dCellW; }
	inline double dCellH() { return m_dCellH; }
	inline double dGamma() { return m_dGamma; }
	inline int nCells() { return m_nCells; }
	inline int nRows() { return m_nRows; }
	inline int nCols() { return m_nCols; }
	// Getters for device pointers
	int *devCellNbrs() { return d_pnCellNbrs; }

	//Setters
	void setLx(double dLx);
	void setLy(double dLy);
	void setMinCellD(double dMinCellD);
	void setGamma(double dGamma) { m_dGamma = dGamma; }

	void rescaleL(double dScaler);

	double dArea() { return m_dLx*m_dLy; }
};


#endif /* BOX_H_ */

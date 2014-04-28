/*
 * disk.h
 *
 *  Created on: Feb 25, 2014
 *      Author: marschat
 */

#ifndef DISK_H_
#define DISK_H_

#include <cuda.h>
#include <vector_types.h>
#include "constants.h"

class DiskArray
{
private:
	int m_nDisks;
	double2 *h_pd2Pos;
	double2 *d_pd2Pos;
	double *h_pdR;
	double *d_pdR;

public:
	DiskArray(int nDisks);
	DiskArray(int nDisks, double2 *pd2Pos, double *pdR);
	~DiskArray();

	void cpyAllToDevice();
	void cpyAllToHost();
	void cpyPosToDevice();
	void cpyPosToHost();

	// Getters
	inline int nDisks() { return m_nDisks; }
	inline double2 d2Pos(int idx) { return h_pd2Pos[idx]; }
	inline double dX(int idx) { return h_pd2Pos[idx].x; }
	inline double dY(int idx) { return h_pd2Pos[idx].y; }
	inline double dR(int idx) { return h_pdR[idx]; }
	// Getters for device pointers
	inline double2 *devPos() { return d_pd2Pos; }
	inline double *devR() { return d_pdR; }

	// Setters
	inline void setPos(int idx, double2 d2Pos) { h_pd2Pos[idx] = d2Pos; }
	inline void setX(int idx, double dX) { h_pd2Pos[idx].x = dX; }
	inline void setY(int idx, double dY) { h_pd2Pos[idx].y = dY; }
	inline void setR(int idx, double dR) { h_pdR[idx] = dR; }

	double dArea();
};

class IdenticalDiskArray
{
private:
	int m_nDisks;
	double2 *h_pd2Pos;
	double2 *d_pd2Pos;
	double m_dR;

public:
	IdenticalDiskArray(int nDisks);
	IdenticalDiskArray(int nDisks, double2 *pd2Pos, double dR);
	~IdenticalDiskArray();

	void cpyAllToDevice();
	void cpyAllToHost();
	void cpyPosToDevice();
	void cpyPosToHost();

	// Getters
	inline int nDisks() { return m_nDisks; }
	inline double2 d2Pos(int idx) { return h_pd2Pos[idx]; }
	inline double dX(int idx) { return h_pd2Pos[idx].x; }
	inline double dY(int idx) { return h_pd2Pos[idx].y; }
	inline double dR() { return m_dR; }
	// Getters for device pointers
	inline double2 *devPos() { return d_pd2Pos; }  // Get device pointer for use in kernel

	// Setters
	inline void setPos(int idx, double2 d2Pos) { h_pd2Pos[idx] = d2Pos; }
	inline void setX(int idx, double dX) { h_pd2Pos[idx].x = dX; }
	inline void setY(int idx, double dY) { h_pd2Pos[idx].y = dY; }
	inline void setR(double dR) { m_dR = dR; }

	double dArea() { return 0.5*m_nDisks*D_PI*m_dR*m_dR; }
};


#endif /* DISK_H_ */

/*
 * spherocyl2d.h
 *
 *  Created on: Feb 25, 2014
 *      Author: marschat
 */

#ifndef SPHEROCYL2D_H_
#define SPHEROCYL2D_H_

#include <cuda.h>
#include <vector_types.h>
#include "constants.h"

class Spherocyl2DArray
{
private:
	int m_nSpherocyls;
	double2 *h_pd2Pos;
	double2 *d_pd2Pos;
	double *h_pdPhi;
	double *d_pdPhi;
	double2 *h_pd2Geo;  // Radius r, and half-shaft length A in double2 struct as (x,y) = (r, A)
	double2 *d_pd2Geo;

public:
	Spherocyl2DArray(int nSpheros);
	Spherocyl2DArray(int nSpheros, double2 *pd2Pos, double *pdPhi, double2 *pd2Geo);
	~Spherocyl2DArray();

	void cpyAllToDevice();
	void cpyAllToHost();
	void cpyPosToDevice();
	void cpyPosToHost();

	// Getters
	inline int nSpherocyls() { return m_nSpherocyls; }
	inline double2 d2Pos(int idx) { return h_pd2Pos[idx]; }
	inline double dX(int idx) { return h_pd2Pos[idx].x; }
	inline double dY(int idx) { return h_pd2Pos[idx].y; }
	inline double dPhi(int idx) { return h_pdPhi[idx]; }
	inline double dR(int idx) { return h_pd2Geo[idx].x; }
	inline double dA(int idx) { return h_pd2Geo[idx].y; }
	// Getters for device pointers
	inline double2 *devPos() { return d_pd2Pos; }
	inline double *devPhi() { return d_pdPhi; }
	inline double2 *devGeo() { return d_pd2Geo; }

	// Setters
	inline void setPos(int idx, double2 d2Pos) { h_pd2Pos[idx] = d2Pos; }
	inline void setX(int idx, double dX) { h_pd2Pos[idx].x = dX; }
	inline void setY(int idx, double dY) { h_pd2Pos[idx].y = dY; }
	inline void setPhi(int idx, double dPhi) { h_pdPhi[idx] = dPhi; }
	inline void setR(int idx, double dR) { h_pd2Geo[idx].x = dR; }
	inline void setA(int idx, double dA) { h_pd2Geo[idx].y = dA; }

	double dArea();
};

class IdenticalSpherocyl2DArray
{
private:
	int m_nSpherocyls;
	double2 *h_pd2Pos;
	double2 *d_pd2Pos;
	double *h_pdPhi;
	double *d_pdPhi;
	double m_dR;
	double m_dA;

public:
	IdenticalSpherocyl2DArray(int nSpheros);
	IdenticalSpherocyl2DArray(int nSpheros, double2 *pd2Pos, double *pdPhi, double dR, double dA);
	~IdenticalSpherocyl2DArray();

	void cpyAllToDevice();
	void cpyAllToHost();
	void cpyPosToDevice();
	void cpyPosToHost();

	// Getters
	inline int nSpherocyls() { return m_nSpherocyls; }
	inline double2 d2Pos(int idx) { return h_pd2Pos[idx]; }
	inline double dX(int idx) { return h_pd2Pos[idx].x; }
	inline double dY(int idx) { return h_pd2Pos[idx].y; }
	inline double dPhi(int idx) { return h_pdPhi[idx]; }
	inline double dR() { return m_dR; }
	inline double dA() { return m_dA; }
	// Getters for device pointers
	inline double2 *devPos() { return d_pd2Pos; }
	inline double *devPhi() { return d_pdPhi; }

	inline void setPos(int idx, double2 d2Pos) { h_pd2Pos[idx] = d2Pos; }
	inline void setX(int idx, double dX) { h_pd2Pos[idx].x = dX; }
	inline void setY(int idx, double dY) { h_pd2Pos[idx].y = dY; }
	inline void setPhi(int idx, double dPhi) { h_pdPhi[idx] = dPhi; }
	inline void setR(double dR) { m_dR = dR; }
	inline void setA(double dA) { m_dA = dA; }

	double dArea() { return m_nSpherocyls*m_dR*(4*m_dA + 0.5*D_PI*m_dR); }
};


#endif /* SPHEROCYL2D_H_ */

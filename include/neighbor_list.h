/*
 * neighbor_list.h
 *
 *  Created on: Mar 17, 2014
 *      Author: marschat
 */

#ifndef NEIGHBOR_LIST_H_
#define NEIGHBOR_LIST_H_

class NeighborList
{
private:
	int m_nParticles;
	int m_nMaxNbrs;
	int *h_pnNbrList;
	int *d_pnNbrList;
public:
	NeighborList(int nParticles, int nMaxNbrs);
	~NeighborList();

	void cpyListToDevice();
	void cpyListToHost();

	inline int nParticles() { return m_nParticles; }
	inline int nMaxNbrs() { return m_nMaxNbrs; }
	int *devNbrList() { return d_pnNbrList; }

	void setParticles(int nParticles);
	void setMaxNbrs(int nMaxNbrs);
};

// TODO:
class CompactNeighborList
{

};


#endif /* NEIGHBOR_LIST_H_ */

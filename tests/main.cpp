/*
 * main.cpp
 *
 *  Created on: Apr 10, 2014
 *      Author: marschat
 */

#include <cuda.h>
#include "granular2d.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <cstdio>

int random_bidisperse_disks(DiskArray &d, Box2D &b, double dSmallR, double dLargeR, int seed = 0)
{
	if (seed)
		srand(seed);
	else
		srand(time(0));

	for (int n = 0; n < d.nDisks(); n++) {
		d.setX(n, b.dLx()*(double)rand()/RAND_MAX);
		d.setY(n, b.dLy()*(double)rand()/RAND_MAX);
		d.setR(n, dSmallR + (n % 2) * (dLargeR - dSmallR));
	}

	return rand();
}

int random_identical_spherocyls(IdenticalSpherocyl2DArray &s, Box2D &b, double dR, double dA, int seed = 0)
{
	if (seed)
		srand(seed);
	else
		srand(time(0));

	for (int n = 0; n < s.nSpherocyls(); n++) {
		s.setX(n, b.dLx()*(double)rand()/RAND_MAX);
		s.setY(n, b.dLy()*(double)rand()/RAND_MAX);
		s.setPhi(n, 2*D_PI*(double)rand()/RAND_MAX);
	}
	s.setR(dR);
	s.setA(dA);

	return rand();
}

int main(int argc, char* argv[])
{
	print_device_info();
	check_error(cudaDeviceReset());

	int nDisks = 4096;
	double dSmallR = 0.5;
	double dLargeR = 0.7;
	double dDiskArea = nDisks*D_PI*0.5*(dSmallR*dSmallR + dLargeR*dLargeR);

	int nSpheros = 1024;
	double dSpheroR = 0.5;
	double dSpheroA = 2.5;
	double dSpheroArea = nSpheros*(4*dSpheroR*dSpheroA + D_PI*dSpheroR*dSpheroR);

	double dMinCellD = fmin(2*(dSpheroR + dSpheroR), 2*dLargeR);
	double dPacking = 0.7;
	double dL = sqrt((dDiskArea+dSpheroArea)/dPacking);

	DiskArray d(nDisks);
	IdenticalSpherocyl2DArray s(nSpheros);
	Box2D b(dL, dL, 0, dMinCellD);
	CellList2D cld(make_int2(b.nCols(),b.nRows()), d.nDisks(), 25);
	CellList2D cls(make_int2(b.nCols(),b.nRows()), s.nSpherocyls(), 15);

	b.cpyCellsToDevice();

	int seed = random_bidisperse_disks(d, b, dSmallR, dLargeR);
	seed = random_identical_spherocyls(s, b, dSpheroR, dSpheroA, seed);

	d.cpyAllToDevice();
	s.cpyAllToDevice();

	KernelConfig dfc_kernel(1,32);
	dfc_kernel.configure(d.nDisks(), 1, 0, 0);
	KernelConfig sfc_kernel(1,32);
	sfc_kernel.configure(s.nSpherocyls(), 1, 0, 0);

	check_last_error("Before finding cells");
	find_cells(cld, b, d, dfc_kernel);
	find_cells(cls, b, s, sfc_kernel);
	check_last_error("While finding cells");

	cls.cpyListToHost();
	d.cpyAllToHost();
	s.cpyAllToHost();
	check_last_error("Copying data from device to host");

	printf("\nSampling %d randomly generated disks:\n", d.nDisks());
	fflush(stdout);
	for (int p = 0; p < d.nDisks(); p+=101) {
		printf("%g %g %g\n", d.dX(p), d.dY(p), d.dR(p));
	}
	fflush(stdout);

	printf("\nSampling %d randomly generated spherocylinders\nWith radius: %g and half-shaft: %g\n",
			s.nSpherocyls(), s.dR(), s.dA());
	fflush(stdout);
	for (int p = 0; p < s.nSpherocyls(); p+=101) {
		printf("%g %g %g\n", s.dX(p), s.dY(p), s.dPhi(p));
	}
	fflush(stdout);

	return 0;
}


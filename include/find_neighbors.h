/*
 * find_neighbors.h
 *
 *  Created on: Apr 22, 2014
 *      Author: marschat
 */

#ifndef FIND_NEIGHBORS_H_
#define FIND_NEIGHBORS_H_

#include "box.h"
#include "cell_list.h"
#include "disk.h"
#include "spherocyl2d.h"
#include "neighbor_list.h"
#include "device_config.h"

void find_cells(CellList2D &cl, Box2D &b, DiskArray &d, KernelConfig k);
void find_cells(CellList2D &cl, Box2D &b, IdenticalDiskArray &d, KernelConfig k);
void find_cells(CellList2D &cl, Box2D &b, Spherocyl2DArray &s, KernelConfig k);
void find_cells(CellList2D &cl, Box2D &b, IdenticalSpherocyl2DArray &s, KernelConfig k);

#endif /* FIND_NEIGHBORS_H_ */

/*
 * error_check.h
 *
 *  Created on: Dec 2, 2013
 *      Author: marschat
 */

#ifndef ERROR_CHECK_H_
#define ERROR_CHECK_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

void check_last_error(const char *msg = "");

void check_error(cudaError_t err, const char *msg = "");


#endif /* ERROR_CHECK_H_ */

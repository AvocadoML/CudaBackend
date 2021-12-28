/*
 * utilities.hpp
 *
 *  Created on: Sep 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <avocado/cuda_backend.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <algorithm>

const int cuda_error_offset = 1000000;
const int cublas_error_offset = 2000000;

template<unsigned int maxBlocks>
unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
{
	return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
}

static avocado::backend::avStatus_t convertStatus(cudaError_t err) noexcept
{
	return static_cast<avocado::backend::avStatus_t>(static_cast<int>(cuda_error_offset + err));
}

static avocado::backend::avStatus_t convertStatus(cublasStatus_t err) noexcept
{
	return static_cast<avocado::backend::avStatus_t>(static_cast<int>(cublas_error_offset + err));
}

static avocado::backend::avStatus_t checkForErrors() noexcept
{
	return convertStatus(cudaGetLastError());
}

#endif /* UTILITIES_H_ */

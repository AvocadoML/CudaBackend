/*
 * utilities.hpp
 *
 *  Created on: Sep 23, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <algorithm>
#include "../include/CudaBackend/cuda_backend.h"

template<unsigned int maxBlocks>
unsigned int gridSize(unsigned int problemSize, unsigned int blockSize) noexcept
{
	return std::min(maxBlocks, (problemSize + blockSize - 1) / blockSize);
}

static avocado::backend::avStatus_t convertStatus(cudaError_t err) noexcept
{
	switch (err)
	{
		case cudaSuccess:
			return avocado::backend::AVOCADO_STATUS_SUCCESS;
		case cudaErrorMemoryAllocation:
			return avocado::backend::AVOCADO_STATUS_ALLOC_FAILED;
		case cudaErrorInsufficientDriver:
			return avocado::backend::AVOCADO_STATUS_INSUFFICIENT_DRIVER;
		default:
			return avocado::backend::AVOCADO_STATUS_EXECUTION_FAILED;
	}
}

static avocado::backend::avStatus_t convertStatus(cublasStatus_t err) noexcept
{
	switch (err)
	{
		case CUBLAS_STATUS_SUCCESS:
			return avocado::backend::AVOCADO_STATUS_SUCCESS;
		case CUBLAS_STATUS_ALLOC_FAILED:
			return avocado::backend::AVOCADO_STATUS_ALLOC_FAILED;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return avocado::backend::AVOCADO_STATUS_ARCH_MISMATCH;
		case CUBLAS_STATUS_NOT_SUPPORTED:
			return avocado::backend::AVOCADO_STATUS_NOT_SUPPORTED;
		default:
			return avocado::backend::AVOCADO_STATUS_EXECUTION_FAILED;
	}
}

static avocado::backend::avStatus_t checkForErrors() noexcept
{
	return convertStatus(cudaGetLastError());
}

namespace avocado
{
	namespace backend
	{
		int cuda_sm_version(int device) noexcept;
	} /* namespace backend */
} /* namespace avocado */

#endif /* UTILITIES_H_ */

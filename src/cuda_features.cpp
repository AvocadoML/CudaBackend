/*
 * cuda_features.cpp
 *
 *  Created on: Sep 23, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <stddef.h>
#include <cstring>
#include <algorithm>
#include <cassert>

namespace
{
	const char* decode_cublas_status(cublasStatus_t status)
	{
		switch (status)
		{
			case CUBLAS_STATUS_SUCCESS:
				return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED:
				return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED:
				return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE:
				return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH:
				return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR:
				return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED:
				return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR:
				return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED:
				return "CUBLAS_STATUS_NOT_SUPPORTED";
			case CUBLAS_STATUS_LICENSE_ERROR:
				return "CUBLAS_STATUS_LICENSE_ERROR";
			default:
				return "unknown status";
		}
	}
	int compute_capability(const cudaDeviceProp &prop) noexcept
	{
		return prop.major * 10 + prop.minor;
	}
}

namespace avocado
{
	namespace backend
	{
		const char* cudaDecodeStatus(avStatus_t status)
		{
			if (status > cublas_error_offset)
				return decode_cublas_status(static_cast<cublasStatus_t>(status - cublas_error_offset));
			else
			{
				if (status > cuda_error_offset)
					return cudaGetErrorName(static_cast<cudaError_t>(status - cuda_error_offset));
				else
					return "Unknown status.";
			}
		}
		avStatus_t cudaGetNumberOfDevices(int *result)
		{
			cudaError_t err = cudaGetDeviceCount(result);
			if (err != cudaSuccess)
				*result = 0;
			return convertStatus(err);
		}

		avStatus_t cudaGetFeatures(CudaFeatures *result, avDeviceIndex_t deviceIdx)
		{
			assert(result != nullptr);
			std::memset(result, 0, sizeof(CudaFeatures));

			cudaDeviceProp prop;
			cudaError_t status = cudaGetDeviceProperties(&prop, deviceIdx);
			if (status == cudaSuccess)
			{
				std::memcpy(result->name, prop.name, std::min(sizeof(result->name), sizeof(prop.name)));
				result->global_memory = prop.totalGlobalMem;
				result->shared_memory = prop.sharedMemPerMultiprocessor;
				result->sm_count = prop.multiProcessorCount;
				result->major = prop.major;
				result->minor = prop.minor;
				result->supports_dp4a = compute_capability(prop) >= 61;
				result->supports_fp16 = compute_capability(prop) >= 53;
				result->supports_bfloat16 = compute_capability(prop) >= 75;
				result->supports_fp64 = compute_capability(prop) >= 13;
				result->has_tensor_cores = false; // TODO
				return AVOCADO_STATUS_SUCCESS;
			} else
				return AVOCADO_STATUS_INTERNAL_ERROR;
		}
		avStatus_t cudaIsCopyPossible(bool *result, avDeviceIndex_t from, avDeviceIndex_t to)
		{
			assert(result != nullptr);
			if (from == to)
				*result = true;
			else
			{
				int tmp = 0;
				cudaError_t status = cudaDeviceCanAccessPeer(&tmp, to, from);
				if (status == cudaSuccess)
					*result = static_cast<bool>(tmp);
				else
					*result = false;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

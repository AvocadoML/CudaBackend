/*
 * cuda_properties.cpp
 *
 *  Created on: Sep 23, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <stddef.h>
#include <cstring>
#include <algorithm>
#include <vector>
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
//			if (status > cublas_error_offset)
//				return decode_cublas_status(static_cast<cublasStatus_t>(status - cublas_error_offset));
//			else
//			{
//				if (status > cuda_error_offset)
//					return cudaGetErrorName(static_cast<cudaError_t>(status - cuda_error_offset));
//				else
			return "Unknown status.";
//			}
		}
		avStatus_t cudaGetNumberOfDevices(int *result)
		{
			cudaError_t err = cudaGetDeviceCount(result);
			if (err != cudaSuccess)
				*result = 0;
			return convertStatus(err);
		}

		avStatus_t cudaGetDeviceProperty(avDeviceIndex_t index, avDeviceProperty_t propertyName, void *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;

			cudaDeviceProp prop;
			cudaError_t status = cudaGetDeviceProperties(&prop, index);
			if (status != cudaSuccess)
				return AVOCADO_STATUS_INTERNAL_ERROR;

			switch (propertyName)
			{
				case AVOCADO_DEVICE_NAME:
					std::memcpy(result, prop.name, std::min(256ul, sizeof(prop.name)));
					break;
				case AVOCADO_DEVICE_PROCESSOR_COUNT:
					reinterpret_cast<int32_t*>(result)[0] = prop.multiProcessorCount;
					break;
				case AVOCADO_DEVICE_MEMORY:
					reinterpret_cast<int64_t*>(result)[0] = prop.totalGlobalMem;
					break;
				case AVOCADO_DEVICE_SUPPORTS_HALF_PRECISION:
					reinterpret_cast<bool*>(result)[0] = compute_capability(prop) >= 53;
					break;
				case AVOCADO_DEVICE_SUPPORTS_BFLOAT16:
					reinterpret_cast<bool*>(result)[0] = true; //compute_capability(prop) >= 75;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SINGLE_PRECISION:
					reinterpret_cast<bool*>(result)[0] = true;
					break;
				case AVOCADO_DEVICE_SUPPORTS_DOUBLE_PRECISION:
					reinterpret_cast<bool*>(result)[0] = compute_capability(prop) >= 13;
					break;
				case AVOCADO_DEVICE_SUPPORTS_SSE:
				case AVOCADO_DEVICE_SUPPORTS_SSE2:
				case AVOCADO_DEVICE_SUPPORTS_SSE3:
				case AVOCADO_DEVICE_SUPPORTS_SSSE3:
				case AVOCADO_DEVICE_SUPPORTS_SSE41:
				case AVOCADO_DEVICE_SUPPORTS_SSE42:
				case AVOCADO_DEVICE_SUPPORTS_AVX:
				case AVOCADO_DEVICE_SUPPORTS_AVX2:
				case AVOCADO_DEVICE_SUPPORTS_AVX512_F:
				case AVOCADO_DEVICE_SUPPORTS_AVX512_VL_BW_DQ:
					reinterpret_cast<bool*>(result)[0] = false;
					break;
				case AVOCADO_DEVICE_SUPPORTS_DP4A:
					reinterpret_cast<bool*>(result)[0] = compute_capability(prop) >= 61;
					break;
				case AVOCADO_DEVICE_ARCH_MAJOR:
					reinterpret_cast<int32_t*>(result)[0] = prop.major;
					break;
				case AVOCADO_DEVICE_ARCH_MINOR:
					reinterpret_cast<int32_t*>(result)[0] = prop.minor;
					break;
				case AVOCADO_DEVICE_SUPPORTS_TENSOR_CORES:
					reinterpret_cast<bool*>(result)[0] = false; // TODO
					break;
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaIsCopyPossible(avDeviceIndex_t from, avDeviceIndex_t to, bool *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;

			if (from == to)
				result[0] = true;
			else
			{
				int tmp = 0;
				cudaError_t status = cudaDeviceCanAccessPeer(&tmp, to, from);
				if (status == cudaSuccess)
					result[0] = static_cast<bool>(tmp);
				else
					result[0] = false;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		int cuda_sm_version(int device) noexcept
		{
			static const std::vector<int> version = []()
			{
				int number;
				cudaGetNumberOfDevices(&number);
				std::vector<int> result;
				for (int i = 0; i < number; i++)
				{
					cudaDeviceProp prop;
					cudaError_t status = cudaGetDeviceProperties(&prop, i);
					if (status != cudaSuccess)
					result.push_back(0);
					else
					result.push_back(compute_capability(prop));
				}
				return result;
			}();
			return version.at(device);
		}
	} /* namespace backend */
} /* namespace avocado */

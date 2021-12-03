/*
 * memory.cu
 *
 *  Created on: Sep 22, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include "context.hpp"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstring>
#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	__global__ void kernel_setall(T *ptr, avSize_t length, T value)
	{
		for (avSize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			ptr[i] = value;
	}
	template<typename T>
	avStatus_t setall_launcher(cudaStream_t stream, void *dst, avSize_t dstSize, const void *value)
	{
		const avSize_t length = dstSize / sizeof(T);
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(length, blockDim.x);

		T v;
		std::memcpy(&v, value, sizeof(T));
		kernel_setall<<<gridDim, blockDim, 0, stream>>>(reinterpret_cast<T*>(dst), length, v);
		return checkForErrors();
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaAllocateMemory(void **ptr, size_t count, avDeviceIndex_t deviceIdx)
		{
			if (ptr == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status = cudaSetDevice(deviceIdx);
			if (status != cudaSuccess)
				return convertStatus(status);
			status = cudaMalloc(ptr, count);
			return convertStatus(status);;
		}
		avStatus_t cudaFreeMemory(void *ptr)
		{
			cudaError_t status = cudaSuccess;
			if (ptr != nullptr)
				status = cudaFree(ptr);
			return convertStatus(status);;
		}
		avStatus_t cudaPageLock(void *ptr, size_t count)
		{
			if (ptr == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status = cudaHostRegister(ptr, count, 0);
			return convertStatus(status);;
		}
		avStatus_t cudaPageUnlock(void *ptr)
		{
			if (ptr == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status = cudaHostUnregister(ptr);
			return convertStatus(status);;
		}
		avStatus_t cudaClearMemory(avContext_t context, void *dst, size_t dstSize)
		{
			if (dst == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			if (context == nullptr)
				status = cudaMemset(dst, 0, dstSize);
			else
				status = cudaMemsetAsync(dst, 0, dstSize, get_stream(context));
			return convertStatus(status);;
		}
		avStatus_t cudaSetMemory(avContext_t context, void *dst, size_t dstSize, const void *pattern, size_t patternSize)
		{
			if (dst == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (pattern == nullptr)
				return cudaClearMemory(context, dst, dstSize);

			if (dstSize % patternSize != 0)
				return AVOCADO_STATUS_BAD_PARAM;
			switch (patternSize)
			{
				case 1:
					return setall_launcher<int8_t>(get_stream(context), dst, dstSize, pattern);
				case 2:
					return setall_launcher<int16_t>(get_stream(context), dst, dstSize, pattern);
				case 4:
					return setall_launcher<int32_t>(get_stream(context), dst, dstSize, pattern);
				case 8:
					return setall_launcher<int2>(get_stream(context), dst, dstSize, pattern);
				case 16:
					return setall_launcher<int4>(get_stream(context), dst, dstSize, pattern);
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
		}
		avStatus_t cudaCopyMemoryToCPU(avContext_t context, void *dst, const void *src, avSize_t count)
		{
			if (dst == nullptr or src == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			if (context == nullptr)
				status = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
			else
				status = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, get_stream(context));
			return convertStatus(status);
		}
		avStatus_t cudaCopyMemoryFromCPU(avContext_t context, void *dst, const void *src, avSize_t count)
		{
			if (dst == nullptr or src == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			if (context == nullptr)
				status = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
			else
				status = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, get_stream(context));
			return convertStatus(status);;
		}
		avStatus_t cudaCopyMemory(avContext_t context, void *dst, const void *src, avSize_t count)
		{
			if (dst == nullptr or src == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			if (context == nullptr)
				status = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice);
			else
				status = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, get_stream(context));
			return convertStatus(status);;
		}
	} /* namespace backend */
} /* namespace avocado */

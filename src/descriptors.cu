/*
 * descriptors.cu
 *
 *  Created on: Dec 22, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "utilities.hpp"

#include <cstring>
#include <memory>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	__global__ void kernel_setall(T *ptr, int length, T value)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			ptr[i] = value;
	}
	template<typename T>
	avStatus_t setall_launcher(cudaStream_t stream, T *dst, int dstSize, const void *value)
	{
		const int length = dstSize / sizeof(T);
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(length, blockDim.x);

		T v;
		std::memcpy(&v, value, sizeof(T));
		kernel_setall<<<gridDim, blockDim, 0, stream>>>(dst, length, v);
		return checkForErrors();
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace avocado::backend::BACKEND_NAMESPACE;

		avStatus_t cudaCreateMemoryDescriptor(avMemoryDescriptor_t *result, avDeviceIndex_t deviceIndex, av_int64 sizeInBytes)
		{
			return create_descriptor<MemoryDescriptor>(result, sizeInBytes, deviceIndex);
		}
		avStatus_t cudaCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, av_int64 sizeInBytes, av_int64 offsetInBytes)
		{
			return create_descriptor<MemoryDescriptor>(result, getMemory(desc), sizeInBytes, offsetInBytes);
		}
		avStatus_t cudaDestroyMemoryDescriptor(avMemoryDescriptor_t desc)
		{
			return destroy_descriptor<MemoryDescriptor>(desc);
		}
		avStatus_t cudaSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, av_int64 dstSize, const void *pattern,
				av_int64 patternSize)
		{
			if (not same_device_type(context, dst))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
			if (getPointer(dst) == nullptr)
			{
				if (dstSize != 0)
					return AVOCADO_STATUS_BAD_PARAM;
				else
					return AVOCADO_STATUS_SUCCESS;
			}
			try
			{
				getContext(context).setDevice();
				if (pattern == nullptr)
				{
					cudaError_t status = cudaGetLastError();
					if (isDefault(context))
						status = cudaMemset(getPointer<int8_t>(dst) + dstOffset, 0, dstSize);
					else
						status = cudaMemsetAsync(getPointer<int8_t>(dst) + dstOffset, 0, dstSize, getContext(context).getStream());
					return convertStatus(status);
				}

				if (dstSize % patternSize != 0 or dstOffset % patternSize != 0)
					return AVOCADO_STATUS_BAD_PARAM;
				switch (patternSize)
				{
					case 1:
						return setall_launcher(getContext(context).getStream(), getPointer<int8_t>(dst) + dstOffset, dstSize, pattern);
					case 2:
						return setall_launcher(getContext(context).getStream(), getPointer<int16_t>(dst) + dstOffset / 2, dstSize, pattern);
					case 4:
						return setall_launcher(getContext(context).getStream(), getPointer<int32_t>(dst) + dstOffset / 4, dstSize, pattern);
					case 8:
						return setall_launcher(getContext(context).getStream(), getPointer<int2>(dst) + dstOffset / 8, dstSize, pattern);
					case 16:
						return setall_launcher(getContext(context).getStream(), getPointer<int4>(dst) + dstOffset / 16, dstSize, pattern);
					default:
						return AVOCADO_STATUS_BAD_PARAM;
				}
			} catch (std::exception &e)
			{
			}
			return AVOCADO_STATUS_INTERNAL_ERROR;
		}
		avStatus_t cudaCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, const avMemoryDescriptor_t src,
				av_int64 srcOffset, av_int64 count)
		{
			if (not same_device_type(context, dst, src))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;
			try
			{
				getContext(context).setDevice();
				bool is_direct_copy_possible;
				cudaIsCopyPossible(getMemory(src).device(), getMemory(dst).device(), &is_direct_copy_possible);
				if (is_direct_copy_possible) // can use peer-to-peer copy
				{
					cudaError_t status;
					if (isDefault(context))
						status = cudaMemcpy(getPointer<int8_t>(dst) + dstOffset, getPointer<int8_t>(src) + srcOffset, count, cudaMemcpyDeviceToDevice);
					else
						status = cudaMemcpyAsync(getPointer<int8_t>(dst) + dstOffset, getPointer<int8_t>(src) + srcOffset, count, cudaMemcpyDeviceToDevice,
								getContext(context).getStream());
					return convertStatus(status);
				}
				else // must use intermediate host buffer
				{
					std::unique_ptr<int8_t[]> buffer = std::make_unique<int8_t[]>(count);
					avStatus_t status = cudaCopyMemoryToHost(context, buffer.get(), src, srcOffset, count);
					if (status != AVOCADO_STATUS_SUCCESS)
						return status;
					status = cudaCopyMemoryFromHost(context, dst, dstOffset, buffer.get(), count);
					if (status != AVOCADO_STATUS_SUCCESS)
						return status;
				}
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaCopyMemoryToHost(avContextDescriptor_t context, void *dst, const avMemoryDescriptor_t src, av_int64 srcOffset, av_int64 bytes)
		{
			if (dst == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			try
			{
				getContext(context).setDevice();
				if (isDefault(context))
					status = cudaMemcpy(dst, getPointer<int8_t>(src) + srcOffset, bytes, cudaMemcpyDeviceToHost);
				else
					status = cudaMemcpyAsync(dst, getPointer<int8_t>(src) + srcOffset, bytes, cudaMemcpyDeviceToHost, getContext(context).getStream());
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return convertStatus(status);
		}
		avStatus_t cudaCopyMemoryFromHost(avContextDescriptor_t context, avMemoryDescriptor_t dst, av_int64 dstOffset, const void *src, av_int64 bytes)
		{
			if (src == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status;
			try
			{
				getContext(context).setDevice();
				if (isDefault(context))
					status = cudaMemcpy(getPointer<int8_t>(dst) + dstOffset, src, bytes, cudaMemcpyHostToDevice);
				else
					status = cudaMemcpyAsync(getPointer<int8_t>(dst) + dstOffset, src, bytes, cudaMemcpyHostToDevice, getContext(context).getStream());
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return convertStatus(status);
		}
		avStatus_t cudaPageLock(void *ptr, av_int64 count)
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

		int cudaGetNumberOfDevices()
		{
			return getNumberOfDevices();
		}

		avStatus_t cudaCreateContextDescriptor(avContextDescriptor_t *result, avDeviceIndex_t deviceIndex)
		{
			return create_descriptor<ContextDescriptor>(result, deviceIndex);
		}
		avStatus_t cudaDestroyContextDescriptor(avContextDescriptor_t desc)
		{
			return destroy_descriptor<ContextDescriptor>(desc);
		}
		avContextDescriptor_t cudaGetDefaultContext(avDeviceIndex_t deviceIndex)
		{
			if (deviceIndex >= 0 and deviceIndex < cudaGetNumberOfDevices())
				return createDescriptor(deviceIndex, ContextDescriptor::descriptor_type);
			else
				return static_cast<avContextDescriptor_t>(-1);
		}
		avStatus_t cudaSynchronizeWithContext(avContextDescriptor_t context)
		{
			try
			{
				getContext(context).setDevice();
				cudaError_t status = cudaStreamSynchronize(getContext(context).getStream());
				if (status != cudaSuccess)
					return AVOCADO_STATUS_INTERNAL_ERROR;
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaIsContextReady(avContextDescriptor_t context, bool *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getContext(context).setDevice();
				cudaError_t status = cudaStreamQuery(getContext(context).getStream());
				if (status == cudaSuccess)
					result[0] = true;
				else
				{
					if (status == cudaErrorNotReady)
						result[0] = false;
					else
					{
						result[0] = false;
						return convertStatus(status);
					}
				}
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaCreateTensorDescriptor(avTensorDescriptor_t *result)
		{
			return create_descriptor<TensorDescriptor>(result, std::initializer_list<int> { }, AVOCADO_DTYPE_UNKNOWN);
		}
		avStatus_t cudaDestroyTensorDescriptor(avTensorDescriptor_t desc)
		{
			return destroy_descriptor<TensorDescriptor>(desc);
		}
		avStatus_t cudaSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[])
		{
			if (nbDims < 0 or nbDims > AVOCADO_MAX_TENSOR_DIMENSIONS)
				return AVOCADO_STATUS_BAD_PARAM;
			if (dimensions == nullptr and nbDims != 0)
				return AVOCADO_STATUS_BAD_PARAM;

			try
			{
				getTensor(desc).set(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[])
		{
			try
			{
				getTensor(desc).get(dtype, nbDims, dimensions);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaCreateConvolutionDescriptor(avConvolutionDescriptor_t *result)
		{
			return create_descriptor<ConvolutionDescriptor>(result);
		}
		avStatus_t cudaDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc)
		{
			return destroy_descriptor<ConvolutionDescriptor>(desc);
		}
		avStatus_t cudaSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[], const int strides[],
				const int dilation[], int groups, const void *paddingValue)
		{
			try
			{
				getConvolution(desc).set(mode, nbDims, padding, strides, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[], int strides[],
				int dilation[], int *groups, void *paddingValue)
		{
			try
			{
				getConvolution(desc).get(mode, nbDims, padding, strides, dilation, groups, paddingValue);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaCreateOptimizerDescriptor(avOptimizerDescriptor_t *result)
		{
			return create_descriptor<OptimizerDescriptor>(result);
		}
		avStatus_t cudaDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc)
		{
			return destroy_descriptor<OptimizerDescriptor>(desc);
		}
		avStatus_t cudaSetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t type, av_int64 steps, double learningRate,
				const double coefficients[], const bool flags[])
		{
			try
			{
				getOptimizer(desc).set(type, steps, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaGetOptimizerDescriptor(avOptimizerDescriptor_t desc, avOptimizerType_t *type, av_int64 *steps, double *learningRate,
				double coefficients[], bool flags[])
		{
			try
			{
				getOptimizer(desc).get(type, steps, learningRate, coefficients, flags);
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, av_int64 *result)
		{
			if (result == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				getOptimizer(desc).get_workspace_size(result, getTensor(wDesc));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_INTERNAL_ERROR;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */


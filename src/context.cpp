/*
 * context.cpp
 *
 *  Created on: Sep 23, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include "context.hpp"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cassert>
#include <stdexcept>

namespace
{
	using namespace avocado::backend;
#define CHECK_CUDA_STATUS(x) if (x != cudaSuccess) throw std::runtime_error("");
#define CHECK_CUBLAS_STATUS(x) if (x != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("");

	struct CudaContext
	{
		Workspace workspace;
		int device_index;
		cudaStream_t stream;
		cublasHandle_t handle;

		CudaContext(int index) :
				device_index(index)
		{
			cudaError_t status = cudaSetDevice(index);
			CHECK_CUDA_STATUS(status)
			cublasStatus_t err = cublasCreate_v2(&handle);
			CHECK_CUBLAS_STATUS(err)
			status = cudaStreamCreate(&stream);
			CHECK_CUDA_STATUS(status)
			err = cublasSetStream_v2(handle, stream);
			CHECK_CUBLAS_STATUS(err)
		}
		~CudaContext() noexcept
		{
			cublasStatus_t err = cublasDestroy_v2(handle);
			assert(err == CUBLAS_STATUS_SUCCESS);
			if (stream != nullptr)
			{
				cudaError_t status = cudaStreamDestroy(stream);
				assert(status == cudaSuccess);
			}
		}
		CudaContext(const CudaContext &other) = delete;
		CudaContext(CudaContext &&other) = delete;
		CudaContext& operator=(const CudaContext &other) = delete;
		CudaContext& operator=(CudaContext &&other) = delete;
	};
	CudaContext* get_instance(avContext_t context)
	{
		assert(context != nullptr);
		return reinterpret_cast<CudaContext*>(context);
	}
}
namespace avocado
{
	namespace backend
	{
		avStatus_t cudaCreateContext(avContext_t *context, avDeviceIndex_t deviceIdx)
		{
			if (context == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			try
			{
				*context = reinterpret_cast<ContextDescriptor*>(new CudaContext(deviceIdx));
			} catch (std::exception &e)
			{
				return AVOCADO_STATUS_ALLOC_FAILED;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaDestroyContext(avContext_t context)
		{
			if (context != nullptr)
				delete get_instance(context);
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaSynchronizeWithContext(avContext_t context)
		{
			cudaError_t status;
			if (context == nullptr)
				status = cudaDeviceSynchronize();
			else
				status = cudaStreamSynchronize(get_instance(context)->stream);
			return convertStatus(status);
		}
		avStatus_t cudaIsContextReady(avContext_t context, bool *result)
		{
			if (context == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			cudaError_t status = cudaStreamQuery(get_instance(context)->stream);
			if (status == cudaSuccess)
			{
				*result = true;
				return AVOCADO_STATUS_SUCCESS;
			}
			else
			{
				if (status == cudaErrorNotReady)
				{
					*result = false;
					return AVOCADO_STATUS_SUCCESS;
				}
				else
				{
					*result = false;
					return convertStatus(status);
				}
			}
		}
		avStatus_t cudaResizeWorkspace(avContext_t context, avSize_t newSize, bool forceShrink)
		{
			if (context == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if ((newSize < get_instance(context)->workspace.size() and forceShrink) or newSize > get_instance(context)->workspace.size())
				get_instance(context)->workspace = Workspace(newSize);
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaGetWorkspace(avContext_t context, void **ptr, avSize_t *sizeInBytes)
		{
			if (context == nullptr)
				return AVOCADO_STATUS_BAD_PARAM;
			if (ptr != nullptr)
				*ptr = get_instance(context)->workspace.get();
			if (sizeInBytes != nullptr)
				*sizeInBytes = get_instance(context)->workspace.size();
			return AVOCADO_STATUS_SUCCESS;
		}

		int get_device(avContext_t context) noexcept
		{
			assert(context != nullptr);
			return get_instance(context)->device_index;
		}
		cudaStream_t get_stream(avContext_t context) noexcept
		{
			if (context == nullptr)
				return 0;
			else
				return get_instance(context)->stream;
		}
		cublasHandle_t get_handle(avContext_t context) noexcept
		{
			assert(context != nullptr);
			return get_instance(context)->handle;
		}

		Workspace cuda_get_workspace(const avContext_t context, avSize_t bytes)
		{
			assert(context != nullptr);
			if (get_instance(context)->workspace.size() < bytes)
				get_instance(context)->workspace = Workspace(bytes);
			return get_instance(context)->workspace;
		}

	} /* namespace backend */
} /* namespace avocado */

/*
 * losses.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	__device__ void reduce_within_block(T *ptr) noexcept
	{
		assert(ispow2(blockDim.x));
		for (int i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.x < i)
				ptr[threadIdx.x] += ptr[threadIdx.x + i];
			__syncthreads();
		}
	}

	/* Functors for calculating losses and gradients */
	class MeanSquareError
	{
	public:
		template<typename T>
		__device__ T getLoss(T output, T target) const noexcept
		{
			return static_cast<T>(0.5) * square(output - target);
		}
		template<typename T>
		__device__ T getGradient(T output, T target) const noexcept
		{
			return output - target;
		}
	};

	class CrossEntropy
	{
		bool is_fused;
	public:
		__device__ __host__ CrossEntropy(bool isFused) :
				is_fused(isFused)
		{
		}
		template<typename T>
		__device__ T getLoss(T output, T target) const noexcept
		{
			return -(target * safe_log(output) + (one<T>() - target) * safe_log(one<T>() - output));
		}
		template<typename T>
		__device__ T getGradient(T output, T target) const noexcept
		{
			return is_fused ? (output - target) : (output - target) / (eps<T>() + output * (one<T>() - output));
		}
	};

	class KLDivergence
	{
		bool is_fused;
	public:
		__device__ __host__ KLDivergence(bool isFused) :
				is_fused(isFused)
		{
		}
		template<typename T>
		__device__ T getLoss(T output, T target) const noexcept
		{
			return -(target * safe_log(output) + (one<T>() - target) * safe_log(one<T>() - output) - target * safe_log(target)
					- (one<T>() - target) * safe_log(one<T>() - target));
		}
		template<typename T>
		__device__ T getGradient(T output, T target) const noexcept
		{
			return is_fused ? (output - target) : (output - target) / (eps<T>() + output * (one<T>() - output));
		}
	};

	/* Kernels for summing loss over entire array */
	template<typename T, class Function>
	__global__ void kernel_reduce_op_step1(T *dst, const T *output, const T *target, int length, T inv_batch_size, Function fn)
	{
		__shared__ T storage[1024];
		T acc = static_cast<T>(0);
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
			acc += fn.getLoss(output[i], target[i]);
		storage[threadIdx.x] = acc;

		__syncthreads();
		reduce_within_block(storage);
		if (threadIdx.x == 0)
			dst[blockIdx.x] = storage[0] * inv_batch_size;
	}
	template<typename T>
	__global__ void kernel_reduce_op_step2(T *dst)
	{
		reduce_within_block(dst);
	}
	template<typename T, class Function>
	__host__ void launch_reduce_op(cudaStream_t stream, const T *output, const T *target, T *workspace, int length, T inv_batch_size, Function fn) noexcept
	{
		dim3 gridDim(gridSize<1024>(length, 1024));
		kernel_reduce_op_step1<<<gridDim, 1024, 0, stream>>>(workspace, output, target, length, inv_batch_size, fn);
		if (gridDim.x > 1)
			kernel_reduce_op_step2<<<1, 1024, 0, stream>>>(workspace);
	}

	/* Kernels for applying an operation to all elements (used for gradient calculation) */
	template<typename T, class Function>
	__global__ void kernel_pointwise_op(T *gradient, const T *output, const T *target, int length, T inv_batch_size, Function fn)
	{
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += gridDim.x * blockDim.x)
			gradient[i] = inv_batch_size * fn.getGradient(output[i], target[i]);
	}
	template<typename T, class Function>
	__host__ void launch_pointwise_op(cudaStream_t stream, T *gradient, const T *output, const T *target, int length, T inv_batch_size, Function fn) noexcept
	{
		dim3 gridDim(gridSize<1024>(length, 256));
		kernel_pointwise_op<<<gridDim, 256, 0, stream>>>(gradient, output, target, length, inv_batch_size, fn);
	}

	template<class Function>
	void dispatch_loss(cudaStream_t stream, const mlTensor_t output, const mlTensor_t target, void *workspace, Function fn) noexcept
	{
		switch (output->dtype)
		{
			case DTYPE_FLOAT32:
			{
				launch_reduce_op(stream, constData<float>(output), constData<float>(target), reinterpret_cast<float*>(workspace), volume(output),
						one<float>() / firstDim(output), fn);
				break;
			}
			case DTYPE_FLOAT64:
			{
				launch_reduce_op(stream, constData<double>(output), constData<double>(target), reinterpret_cast<double*>(workspace), volume(output),
						one<double>() / firstDim(output), fn);
				break;
			}
		}
	}
	template<class Function>
	void dispatch_gradient(cudaStream_t stream, mlTensor_t gradient, const mlTensor_t output, const mlTensor_t target, Function fn) noexcept
	{
		switch (output->dtype)
		{
			case DTYPE_FLOAT32:
			{
				launch_pointwise_op(stream, data<float>(gradient), constData<float>(output), constData<float>(target), volume(output),
						one<float>() / firstDim(output), fn);
				break;
			}
			case DTYPE_FLOAT64:
			{
				launch_pointwise_op(stream, data<double>(gradient), constData<double>(output), constData<double>(target), volume(output),
						one<double>() / firstDim(output), fn);
				break;
			}
		}
	}
}

namespace avocado
{
	namespace backend
	{

		mlStatus_t cudaLossFunction(mlContext_t context, mlLossType_t lossType, mlScalar_t result, const mlTensor_t output, const mlTensor_t target)
		{
			void *workspace = cuda_get_workspace(context, dataTypeSize(output->dtype) * volume(output));
			switch (lossType)
			{
				case MEAN_SQUARE_LOSS:
					dispatch_loss(getStream(context), output, target, workspace, MeanSquareError());
					break;
				case CROSS_ENTROPY_LOSS:
					dispatch_loss(getStream(context), output, target, workspace, CrossEntropy(false));
					break;
				case KL_DIVERGENCE_LOSS:
					dispatch_loss(getStream(context), output, target, workspace, KLDivergence(false));
					break;
				default:
					return STATUS_NOT_SUPPORTED;
			}
			cudaCopyMemoryToCPU(context, result->data, workspace, dataTypeSize(result->dtype));
			cudaStreamSynchronize(getStream(context));
			return checkForErrors();
		}
		mlStatus_t cudaLossGradient(mlContext_t context, mlLossType_t lossType, mlTensor_t gradient, const mlTensor_t output, const mlTensor_t target,
				bool fused)
		{
			switch (lossType)
			{
				case MEAN_SQUARE_LOSS:
					dispatch_gradient(getStream(context), gradient, output, target, MeanSquareError());
					break;
				case CROSS_ENTROPY_LOSS:
					dispatch_gradient(getStream(context), gradient, output, target, CrossEntropy(fused));
					break;
				case KL_DIVERGENCE_LOSS:
					dispatch_gradient(getStream(context), gradient, output, target, KLDivergence(fused));
					break;
				default:
					return STATUS_NOT_SUPPORTED;
			}
			cudaStreamSynchronize(getStream(context));
			return checkForErrors();
		}

		avStatus_t cudaLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

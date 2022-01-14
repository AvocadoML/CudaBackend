/*
 * losses.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
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
	template<class Function, typename T, typename U = T>
	__global__ void kernel_pointwise_op(T *gradient, const T *output, const T *target, unsigned int elements, U alpha, U beta, Function fn)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			U tmp = alpha * fn.getGradient(output[i], target[i]);
			if (beta != zero<U>())
				tmp += beta * gradient[i];
			gradient[i] = tmp;
		}
	}

	template<class Function>
	void dispatch_loss(avContextDescriptor_t context, const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem,
			const avMemoryDescriptor_t targetMem, void *result, Function fn) noexcept
	{
		const unsigned int elements = getTensor(outputDesc).volume();
		const unsigned int batch_size = getTensor(outputDesc).firstDim();
		cudaStream_t stream = getContext(context).getStream();

		switch (getTensor(outputDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float * workspace = getContext(context).getWorkspace().data<float>();
				launch_reduce_op(stream, getPointer<float>(outputMem), getPointer<float>(targetMem), workspace, elements, 1.0f / batch_size, fn);
				cudaMemcpyAsync(result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double * workspace = getContext(context).getWorkspace().data<double>();
				launch_reduce_op(stream, getPointer<double>(outputMem), getPointer<double>(targetMem), workspace, elements, 1.0 / batch_size, fn);
				cudaMemcpyAsync(result, workspace, sizeof(double), cudaMemcpyDeviceToHost, stream);
				break;
			}
		}
		cudaStreamSynchronize(stream);
	}
	template<class Function>
	void dispatch_gradient(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem,
			const avMemoryDescriptor_t targetMem, const void *beta, avMemoryDescriptor_t gradientMem, Function fn) noexcept
	{
		const unsigned int elements = getTensor(outputDesc).volume();
		const unsigned int batch_size = getTensor(outputDesc).firstDim();
		cudaStream_t stream = getContext(context).getStream();

		dim3 blockDim(256);
		dim3 gridDim(gridSize<1024>(elements, blockDim.x));

		switch (getTensor(outputDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_pointwise_op<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(gradientMem), getPointer<float>(outputMem),
						getPointer<float>(targetMem), elements, getAlphaValue(alpha) / batch_size, getBetaValue(beta), fn);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_pointwise_op<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(gradientMem), getPointer<double>(outputMem),
						getPointer<double>(targetMem), elements, getAlphaValue<double>(alpha) / batch_size, getBetaValue<double>(beta), fn);
				break;
			}
		}
		cudaStreamSynchronize(stream);
	}
}

namespace avocado
{
	namespace backend
	{

		avStatus_t cudaLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			switch (lossType)
			{
				case AVOCADO_MEAN_SQUARE_LOSS:
					dispatch_loss(context, outputDesc, outputMem, targetMem, result, MeanSquareError());
					break;
				case AVOCADO_CROSS_ENTROPY_LOSS:
					dispatch_loss(context, outputDesc, outputMem, targetMem, result, CrossEntropy(false));
					break;
				case AVOCADO_KL_DIVERGENCE_LOSS:
					dispatch_loss(context, outputDesc, outputMem, targetMem, result, KLDivergence(false));
					break;
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
			return checkForErrors();
		}
		avStatus_t cudaLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, const void *beta,
				const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem, bool isFused)
		{
			switch (lossType)
			{
				case AVOCADO_MEAN_SQUARE_LOSS:
					dispatch_gradient(context, alpha, outputDesc, outputMem, targetMem, beta, gradientMem, MeanSquareError());
					break;
				case AVOCADO_CROSS_ENTROPY_LOSS:
					dispatch_gradient(context, alpha, outputDesc, outputMem, targetMem, beta, gradientMem, CrossEntropy(isFused));
					break;
				case AVOCADO_KL_DIVERGENCE_LOSS:
					dispatch_gradient(context, alpha, outputDesc, outputMem, targetMem, beta, gradientMem, KLDivergence(isFused));
					break;
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
			return checkForErrors();
		}

	} /* namespace backend */
} /* namespace avocado */

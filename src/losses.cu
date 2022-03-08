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
			return -(target * safe_log(output) + (scalar_one<T>() - target) * safe_log(scalar_one<T>() - output));
		}
		template<typename T>
		__device__ T getGradient(T output, T target) const noexcept
		{
			return is_fused ? (output - target) : (output - target) / (scalar_eps<T>() + output * (scalar_one<T>() - output));
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
			CrossEntropy ce(is_fused);
			return ce.getLoss(output, target) - ce.getLoss(target, target);
		}
		template<typename T>
		__device__ T getGradient(T output, T target) const noexcept
		{
			CrossEntropy ce(is_fused);
			return ce.getGradient(output, target);
		}
	};

	/* Kernels for summing loss over entire array */
	template<typename T, class Function>
	__global__ void kernel_reduce_op_step1(T *dst, const T *output, const T *target, int length, Function fn)
	{
		assert(blockDim.x == 1024);
		__shared__ T storage[1024];
		T acc = static_cast<T>(0);
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < length; i += blockDim.x * gridDim.x)
			acc += fn.getLoss(output[i], target[i]);
		storage[threadIdx.x] = acc;

		__syncthreads();
		reduce_within_block(storage);
		if (threadIdx.x == 0)
			dst[blockIdx.x] = storage[0];
	}
	template<typename T>
	__global__ void kernel_reduce_op_step2(T *dst)
	{
		reduce_within_block(dst);
	}
	template<typename T, class Function>
	__host__ void launch_reduce_op(cudaStream_t stream, const T *output, const T *target, T *workspace, int length, Function fn) noexcept
	{
		dim3 gridDim(gridSize<1024>(length, 1024));
		kernel_reduce_op_step1<<<gridDim, 1024, 0, stream>>>(workspace, output, target, length, fn);
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
			if (beta != scalar_zero<U>())
				tmp += beta * gradient[i];
			gradient[i] = tmp;
		}
	}

	template<class Function>
	void dispatch_loss(avContextDescriptor_t context, const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem,
			const avMemoryDescriptor_t targetMem, void *result, Function fn) noexcept
	{
		const unsigned int elements = cuda::getTensor(outputDesc).volume();
		const unsigned int batch_size = cuda::getTensor(outputDesc).firstDim();
		cudaStream_t stream = cuda::getContext(context).getStream();

		switch (cuda::getTensor(outputDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float *workspace = cuda::getContext(context).getWorkspace().data<float>();
				launch_reduce_op(stream, cuda::getPointer<float>(outputMem), cuda::getPointer<float>(targetMem), workspace, elements, fn);
				cudaMemcpyAsync(result, workspace, sizeof(float), cudaMemcpyDeviceToHost, stream);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double *workspace = cuda::getContext(context).getWorkspace().data<double>();
				launch_reduce_op(stream, cuda::getPointer<double>(outputMem), cuda::getPointer<double>(targetMem), workspace, elements, fn);
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
		const unsigned int elements = cuda::getTensor(outputDesc).volume();
		cudaStream_t stream = cuda::getContext(context).getStream();

		dim3 blockDim(256);
		dim3 gridDim(gridSize<1024>(elements, blockDim.x));

		switch (cuda::getTensor(outputDesc).dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				kernel_pointwise_op<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(gradientMem), cuda::getPointer<float>(outputMem),
						cuda::getPointer<float>(targetMem), elements, cuda::getAlphaValue(alpha), cuda::getBetaValue(beta), fn);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				kernel_pointwise_op<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(gradientMem), cuda::getPointer<double>(outputMem),
						cuda::getPointer<double>(targetMem), elements, cuda::getAlphaValue<double>(alpha), cuda::getBetaValue<double>(beta), fn);
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
			cuda::getContext(context).setDevice();
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
			cuda::getContext(context).setDevice();
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

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
#include "reduction_utils.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	using namespace avocado::backend;

	/* Functors for calculating losses and gradients */
	class MeanSquareError
	{
	public:
		template<typename T>
		__device__ numbers::Number<T> getLoss(numbers::Number<T> output, numbers::Number<T> target) const noexcept
		{
			return static_cast<T>(0.5) * square(output - target);
		}
		template<typename T>
		__device__ numbers::Number<T> getGradient(numbers::Number<T> output, numbers::Number<T> target) const noexcept
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
		__device__ numbers::Number<T> getLoss(numbers::Number<T> output, numbers::Number<T> target) const noexcept
		{
			return -(target * safe_log(output) + (numbers::one<T>() - target) * safe_log(numbers::one<T>() - output));
		}
		template<typename T>
		__device__ numbers::Number<T> getGradient(numbers::Number<T> output, numbers::Number<T> target) const noexcept
		{
			if (is_fused)
				return (output - target);
			else
				return (output - target) / (numbers::epsilon<T>() + output * (numbers::one<T>() - output));
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
		__device__ numbers::Number<T> getLoss(numbers::Number<T> output, numbers::Number<T> target) const noexcept
		{
			CrossEntropy ce(is_fused);
			return ce.getLoss(output, target) - ce.getLoss(target, target);
		}
		template<typename T>
		__device__ numbers::Number<T> getGradient(numbers::Number<T> output, numbers::Number<T> target) const noexcept
		{
			CrossEntropy ce(is_fused);
			return ce.getGradient(output, target);
		}
	};

	/* Kernels for summing loss over entire array */
	template<typename T, class Function>
	__global__ void kernel_reduce_op_step1(T *dst, const T *output, const T *target, int elements, Function fn)
	{
		assert(blockDim.x == 1024);
		__shared__ ReduceAdd<T> storage[1024];

		ReduceAdd<T> acc;
		acc.init();
		for (uint32_t i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> _output(output + i, elements - i);
			numbers::Number<T> _target(target + i, elements - i);
			acc.accumulate(fn.getLoss(_output, _target));
		}
		storage[threadIdx.x] = acc;

		__syncthreads();
		block_linear_reducion(storage);
		if (threadIdx.x == 0)
		{
			numbers::Number<T> tmp = (numbers::Number<T>) storage[0];
			tmp.store(dst + numbers::length<T>() * blockIdx.x);
		}
	}
	template<typename T>
	__global__ void kernel_reduce_op_step2(T *dst)
	{
		assert(blockDim.x <= 1024);
		__shared__ ReduceAdd<T> storage[1024];
		for (int i = threadIdx.x; i < 1024; i += blockDim.x)
			storage[i].init();
		storage[threadIdx.x] = numbers::Number<T>(dst + numbers::length<T>() * threadIdx.x);
		__syncthreads();
		block_linear_reducion(storage);

		if (threadIdx.x == 0)
		{
			storage[0].horizontal_reduction();
			storage[0].final_action();
			numbers::Number<T> tmp = (numbers::Number<T>) storage[0];
			tmp.store(dst, 1);
		}
	}
	template<typename T, class Function>
	avStatus_t launch_reduce_op(cudaStream_t stream, const T *output, const T *target, cuda::MemoryDescriptor &workspace, int elements, Function fn) noexcept
	{
		assert(output != nullptr);
		assert(target != nullptr);
		dim3 blockDim(1024);
		const int partial_results = round_to_power_of_2(gridSize<1024>(elements, blockDim.x));
		if (workspace.size() < sizeof(T) * partial_results)
			return AVOCADO_STATUS_INTERNAL_ERROR;

		kernel_reduce_op_step1<T, Function> <<<partial_results, blockDim, 0, stream>>>(workspace.data<T>(), output, target, elements, fn);
		kernel_reduce_op_step2<T> <<<1, partial_results, 0, stream>>>(workspace.data<T>());
		return checkForErrors();

//		dim3 blockDim(1024);
//		const int partial_results = round_to_power_of_2(gridSize<1024>(dimensions.first, blockDim.x));
//		if (workspace.size() < sizeof(T) * partial_results)
//			return AVOCADO_STATUS_INTERNAL_ERROR;
//
//		kernel_reduce_linear_1<Op, T> <<<partial_results, blockDim, 0, stream>>>(workspace.data<T>(), input, dimensions.first);
//		kernel_reduce_linear_2<Op, T, U> <<<1, partial_results, 0, stream>>>(output, workspace.data<T>(), alpha, beta);

//		kernel_reduce_op_step1<<<1, 1024, 0, stream>>>(workspace, output, target, length, fn);
//		dim3 gridDim(gridSize<1024>(length, 1024));
//		kernel_reduce_op_step1<<<gridDim, 1024, 0, stream>>>(workspace, output, target, length, fn);
//		if (gridDim.x > 1)
//			kernel_reduce_op_step2<<<1, 1024, 0, stream>>>(workspace);
	}

	/* Kernels for applying an operation to all elements (used for gradient calculation) */
	template<class Function, typename T, typename U = T>
	__global__ void kernel_pointwise_op(T *gradient, const T *output, const T *target, unsigned int elements, U alpha, U beta, Function fn)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			numbers::Number<T> _output(output + i, elements - i);
			numbers::Number<T> _target(target + i, elements - i);
			U tmp = alpha * fn.getGradient(_output, _target);
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
				avStatus_t status = launch_reduce_op(stream, cuda::getPointer<float>(outputMem), cuda::getPointer<float>(targetMem),
						cuda::getContext(context).getWorkspace(), elements, fn);
				cudaMemcpyAsync(result, cuda::getContext(context).getWorkspace().data<float>(), sizeof(float), cudaMemcpyDeviceToHost, stream);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				avStatus_t status = launch_reduce_op(stream, cuda::getPointer<double>(outputMem), cuda::getPointer<double>(targetMem),
						cuda::getContext(context).getWorkspace(), elements, fn);
				cudaMemcpyAsync(result, cuda::getContext(context).getWorkspace().data<double>(), sizeof(double), cudaMemcpyDeviceToHost, stream);
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

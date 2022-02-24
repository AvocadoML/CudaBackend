/*
 * optimizers.cu
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
	__device__ T round_small_to_zero(T x) noexcept
	{
		if (x > -eps<T>() and x < eps<T>())
			return zero<T>();
		else
			return x;
	}

	template<typename T>
	__global__ void kernel_learn_sgd(T *weight, const T *update, T *momentum, unsigned int elements, T learning_rate, T beta, bool use_momentum,
			bool use_nesterov)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			if (use_momentum)
			{
				momentum[i] = beta * momentum[i] - learning_rate * update[i];
				if (use_nesterov)
					weight[i] += beta * momentum[i] - learning_rate * update[i];
				else
					weight[i] += momentum[i];
			}
			else
				weight[i] -= learning_rate * update[i];
			weight[i] = round_small_to_zero(weight[i]);
		}
	}
	template<typename T>
	__global__ void kernel_learn_adam(T *weight, const T *update, T *momentum, T *variance, unsigned int elements, T learning_rate, T beta1, T beta2)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			momentum[i] = momentum[i] * beta1 + update[i] * (one<T>() - beta1);
			variance[i] = variance[i] * beta2 + square(update[i]) * (one<T>() - beta2);
			weight[i] -= momentum[i] * learning_rate / sqrt(variance[i] + eps<T>());
			weight[i] = round_small_to_zero(weight[i]);
		}
	}

	avStatus_t launcher_sgd(const cuda::ContextDescriptor &context, const cuda::OptimizerDescriptor &config, const cuda::TensorDescriptor &wDesc,
			cuda::MemoryDescriptor &wMem, const cuda::MemoryDescriptor &dwMem, cuda::MemoryDescriptor& workspace)
	{
		const avSize_t elements = wDesc.volume();
		bool use_momentum = config.flags[0];
		bool use_nesterov = config.flags[1];
		if (use_momentum)
		{
			if (workspace.size() < elements * cuda::dataTypeSize(wDesc.dtype()))
				return AVOCADO_STATUS_INTERNAL_ERROR;
		}

		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = context.getStream();

		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float beta = config.coef[0];
				float learning_rate = config.learning_rate;
				float *momentum = use_momentum ? nullptr : workspace.data<float>();
				kernel_learn_sgd<<<gridDim, blockDim, 0, stream>>>(wMem.data<float>(), dwMem.data<float>(), momentum, elements, learning_rate, beta,
						use_momentum, use_nesterov);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double beta = config.coef[0];
				double learning_rate = config.learning_rate;
				double *momentum = use_momentum ? nullptr : workspace.data<double>();
				kernel_learn_sgd<<<gridDim, blockDim, 0, stream>>>(wMem.data<double>(), dwMem.data<double>(), momentum, elements, learning_rate, beta,
						use_momentum, use_nesterov);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return checkForErrors();
	}
	avStatus_t launcher_adam(const cuda::ContextDescriptor & context, const cuda::OptimizerDescriptor& config, const cuda::TensorDescriptor & wDesc,
			cuda::MemoryDescriptor & wMem, const cuda::MemoryDescriptor & dwMem, cuda::MemoryDescriptor & workspace)
	{
		const avSize_t elements = wDesc.volume();

		if (workspace.size() < 2 * elements * cuda::dataTypeSize(wDesc.dtype()))
			return AVOCADO_STATUS_INTERNAL_ERROR;

		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = context.getStream();

		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				float beta1 = config.coef[0];
				float beta2 = config.coef[1];
				float learning_rate = config.learning_rate;
				kernel_learn_adam<<<gridDim, blockDim, 0, stream>>>(wMem.data<float>(), dwMem.data<float>(), workspace.data<float>(),
						workspace.data<float>() + elements, elements, learning_rate, beta1, beta2);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				double beta1 = config.coef[0];
				double beta2 = config.coef[1];
				double learning_rate = config.learning_rate;
				kernel_learn_adam<<<gridDim, blockDim, 0, stream>>>(wMem.data<double>(), dwMem.data<double>(), workspace.data<double>(),
						workspace.data<double>() + elements, elements, learning_rate, beta1, beta2);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return checkForErrors();
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const void *alpha, const avTensorDescriptor_t dwDesc,
				const avTensorDescriptor_t dwMem, const void *beta, const avTensorDescriptor_t wDesc, avMemoryDescriptor_t wMem, avMemoryDescriptor_t workspace)
		{
			cuda::getContext(context).setDevice();
			switch (cuda::getOptimizer(config).type)
			{
				case AVOCADO_OPTIMIZER_SGD:
					return launcher_sgd(cuda::getContext(context), cuda::getOptimizer(config), cuda::getTensor(wDesc), cuda::getMemory(wMem),
							cuda::getMemory(dwMem), cuda::getMemory(workspace));
				case AVOCADO_OPTIMIZER_ADAM:
					return launcher_adam(cuda::getContext(context), cuda::getOptimizer(config), cuda::getTensor(wDesc), cuda::getMemory(wMem),
							cuda::getMemory(dwMem), cuda::getMemory(workspace));
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

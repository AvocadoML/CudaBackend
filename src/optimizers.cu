/*
 * optimizers.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	template<typename T>
	__device__ T round_small_to_zero(T x)
	{
		if (x > -scalar_eps<T>() and x < scalar_eps<T>())
			return scalar_zero<T>();
		else
			return x;
	}

	template<typename T>
	__global__ void kernel_learn_sgd(T *weight, const T *update, T *momentum, uint32_t elements, T learning_rate, T beta1, bool use_momentum, bool use_nesterov,
			T alpha, T beta)
	{
		for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			T tmp;
			if (use_momentum)
			{
				momentum[i] = beta1 * momentum[i] - learning_rate * update[i];
				if (use_nesterov)
					tmp = beta1 * momentum[i] - learning_rate * update[i];
				else
					tmp = momentum[i];
			}
			else
				tmp = -learning_rate * update[i];
			weight[i] = round_small_to_zero(alpha * tmp + beta * weight[i]);
		}
	}
	template<typename T>
	__global__ void kernel_learn_adam(T *weight, const T *update, T *momentum, T *variance, uint32_t elements, T learning_rate, T beta1, T beta2, T alpha,
			T beta)
	{
		for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			momentum[i] = momentum[i] * beta1 + update[i] * (scalar_one<T>() - beta1);
			variance[i] = variance[i] * beta2 + square(update[i]) * (scalar_one<T>() - beta2);
			T tmp = -momentum[i] * learning_rate / sqrt(variance[i] + scalar_eps<T>());
			weight[i] = round_small_to_zero(alpha * tmp + beta * weight[i]);
		}
	}

	avStatus_t launcher_sgd(const ContextDescriptor &context, const OptimizerDescriptor &config, const void *alpha, const void *beta,
			const TensorDescriptor &wDesc, MemoryDescriptor &wMem, const MemoryDescriptor &dwMem, MemoryDescriptor &workspace)
	{
		const uint32_t elements = wDesc.volume();
		const bool use_momentum = config.flags[0];
		const bool use_nesterov = config.flags[1];
		if (use_momentum)
		{
			if (workspace.sizeInBytes() < elements * dataTypeSize(wDesc.dtype()))
				return AVOCADO_STATUS_INTERNAL_ERROR;
		}

		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = context.getStream();

		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				const float _alpha = getAlphaValue(alpha);
				const float _beta = getBetaValue(beta);
				const float beta1 = config.coef[0];
				const float learning_rate = config.learning_rate;
				float *momentum = use_momentum ? workspace.data<float>() : nullptr;
				kernel_learn_sgd<<<gridDim, blockDim, 0, stream>>>(wMem.data<float>(), dwMem.data<float>(), momentum, elements, learning_rate, beta1,
						use_momentum, use_nesterov, _alpha, _beta);
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				const double _alpha = getAlphaValue<double>(alpha);
				const double _beta = getBetaValue<double>(beta);
				const double beta1 = config.coef[0];
				const double learning_rate = config.learning_rate;
				double *momentum = use_momentum ? workspace.data<double>() : nullptr;
				kernel_learn_sgd<<<gridDim, blockDim, 0, stream>>>(wMem.data<double>(), dwMem.data<double>(), momentum, elements, learning_rate, beta1,
						use_momentum, use_nesterov, _alpha, _beta);
				break;
			}
			default:
				return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}
		return checkForErrors();
	}
	avStatus_t launcher_adam(const ContextDescriptor &context, OptimizerDescriptor &optimizer, const void *alpha, const void *beta,
			const TensorDescriptor &wDesc, MemoryDescriptor &wMem, const MemoryDescriptor &dwMem, MemoryDescriptor &workspace)
	{
		const uint32_t elements = wDesc.volume();

		if (workspace.sizeInBytes() < 2 * elements * dataTypeSize(wDesc.dtype()))
			return AVOCADO_STATUS_INTERNAL_ERROR;

		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		cudaStream_t stream = context.getStream();

		const int64_t steps = optimizer.steps;
		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
			{
				const float _alpha = getAlphaValue(alpha);
				const float _beta = getBetaValue(beta);
				const float beta1 = optimizer.coef[0];
				const float beta2 = optimizer.coef[1];
				float learning_rate = optimizer.learning_rate;
				if (steps < 10000)
					learning_rate *= std::sqrt(1.0f - std::pow(beta2, steps)) / (1.0f - std::pow(beta1, steps));
				kernel_learn_adam<<<gridDim, blockDim, 0, stream>>>(wMem.data<float>(), dwMem.data<float>(), workspace.data<float>(),
						workspace.data<float>() + elements, elements, learning_rate, beta1, beta2, _alpha, _beta);
				optimizer.steps++;
				break;
			}
			case AVOCADO_DTYPE_FLOAT64:
			{
				const double _alpha = getAlphaValue<double>(alpha);
				const double _beta = getBetaValue<double>(beta);
				const double beta1 = optimizer.coef[0];
				const double beta2 = optimizer.coef[1];
				double learning_rate = optimizer.learning_rate;
				if (steps < 10000)
					learning_rate *= std::sqrt(1.0 - std::pow(beta2, steps)) / (1.0 - std::pow(beta1, steps));
				kernel_learn_adam<<<gridDim, blockDim, 0, stream>>>(wMem.data<double>(), dwMem.data<double>(), workspace.data<double>(),
						workspace.data<double>() + elements, elements, learning_rate, beta1, beta2, _alpha, _beta);
				optimizer.steps++;
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
		using namespace BACKEND_NAMESPACE;

		avStatus_t cudaOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const void *alpha, const avTensorDescriptor_t dwDesc,
				const avTensorDescriptor_t dwMem, const void *beta, const avTensorDescriptor_t wDesc, avMemoryDescriptor_t wMem, avMemoryDescriptor_t workspace)
		{
			getContext(context).setDevice();
			switch (getOptimizer(config).type)
			{
				case AVOCADO_OPTIMIZER_SGD:
					return launcher_sgd(getContext(context), getOptimizer(config), alpha, beta, getTensor(wDesc), getMemory(wMem), getMemory(dwMem),
							getMemory(workspace));
				case AVOCADO_OPTIMIZER_ADAM:
					return launcher_adam(getContext(context), getOptimizer(config), alpha, beta, getTensor(wDesc), getMemory(wMem), getMemory(dwMem),
							getMemory(workspace));
				default:
					return AVOCADO_STATUS_BAD_PARAM;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

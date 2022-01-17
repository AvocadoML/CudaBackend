/*
 * activations.cu
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
	template<class Activation, typename T, typename U = T>
	__global__ void kernel_act_forward(const T *input, T *output, U alpha, U beta, unsigned int elements)
	{
		Activation activation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			T tmp = alpha * activation.forward(input[i]);
			if (beta != zero<U>())
				tmp += beta * output[i];
			output[i] = tmp;
		}
	}
	template<typename T, typename U = T>
	avStatus_t helper_act_forward(cudaStream_t stream, const T *input, T *output, U alpha, U beta, unsigned int elements, avActivationType_t activation)
	{
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_act_forward<ActivationLinear<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_act_forward<ActivationSigmoid<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_act_forward<ActivationTanh<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_act_forward<ActivationRelu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_act_forward<ActivationSelu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_act_forward<ActivationElu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_act_forward<ActivationExponential<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_act_forward<ActivationSoftplus<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_act_forward<ActivationSoftsign<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return checkForErrors();
	}

	template<class Activation, typename T, typename U = T>
	__global__ void kernel_act_backward(T *gradient_prev, const T *gradient_next, const T *output, U alpha, U beta, unsigned int elements)
	{
		Activation activation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			T tmp = alpha * activation.backward(gradient_next[i], output[i]);
			if (beta != zero<U>())
				tmp += beta * gradient_prev[i];
			gradient_prev[i] = tmp;
		}
	}
	template<typename T, typename U = T>
	avStatus_t helper_act_backward(cudaStream_t stream, T *gradient_prev, const T *gradient_next, const T *output, U alpha, U beta, unsigned int elements,
			avActivationType_t activation)
	{
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		switch (activation)
		{
			case AVOCADO_ACTIVATION_LINEAR:
				kernel_act_backward<ActivationLinear<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_act_backward<ActivationSigmoid<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_act_backward<ActivationTanh<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_act_backward<ActivationRelu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_act_backward<ActivationSelu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_act_backward<ActivationElu<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_act_backward<ActivationExponential<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
						elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_act_backward<ActivationSoftplus<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
						elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_act_backward<ActivationSoftsign<T>, T, U> <<<blockDim, gridDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
						elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return checkForErrors();
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			const unsigned int elements = cuda::getTensor(yDesc).volume();
			cudaStream_t stream = cuda::getContext(context).getStream();

			switch (cuda::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return helper_act_forward(stream, cuda::getPointer<float>(xMem), cuda::getPointer<float>(yMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), elements, activation);
				case AVOCADO_DTYPE_FLOAT64:
					return helper_act_forward(stream, cuda::getPointer<double>(xMem), cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha),
							cuda::getBetaValue<double>(beta), elements, activation);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

		avStatus_t cudaActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const unsigned int elements = cuda::getTensor(yDesc).volume();
			cudaStream_t stream = cuda::getContext(context).getStream();

			switch (cuda::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return helper_act_backward(stream, cuda::getPointer<float>(dxMem), cuda::getPointer<float>(dyMem), cuda::getPointer<float>(yMem),
							cuda::getAlphaValue(alpha), cuda::getBetaValue(beta), elements, activation);
				case AVOCADO_DTYPE_FLOAT64:
					return helper_act_backward(stream, cuda::getPointer<double>(dxMem), cuda::getPointer<double>(dyMem), cuda::getPointer<double>(yMem),
							cuda::getAlphaValue<double>(alpha), cuda::getBetaValue<double>(beta), elements, activation);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

	} /* namespace backend */
} /* namespace avocado */

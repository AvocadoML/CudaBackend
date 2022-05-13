/*
 * activations.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "numbers/numbers.cuh"
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
		numbers::Number<T> _alpha(alpha);
		numbers::Number<T> _beta(beta);
		Activation activation;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * gridDim.x * blockDim.x)
		{
			int elements_left = elements - i;
			numbers::Number<T> tmp(input + i, elements_left);

			tmp = _alpha * activation.forward(tmp);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(output + i, elements_left);
			tmp.store(output + i, elements_left);

//			Number<T> tmp = alpha * activation.forward();
//			if (beta != zero<U>())
//				tmp += beta * output[i];
//			output[i] = tmp;
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
				kernel_act_forward<ActivationLinear<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_act_forward<ActivationSigmoid<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_act_forward<ActivationTanh<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_act_forward<ActivationRelu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_act_forward<ActivationSelu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_act_forward<ActivationElu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_act_forward<ActivationExponential<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_act_forward<ActivationSoftplus<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_act_forward<ActivationSoftsign<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return checkForErrors();
	}

	template<class Activation, typename T, typename U = T>
	__global__ void kernel_act_backward(T *gradient_prev, const T *gradient_next, const T *output, U alpha, U beta, unsigned int elements)
	{
		numbers::Number<T> _alpha(alpha);
		numbers::Number<T> _beta(beta);
		Activation activation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			numbers::Number<T> grad(gradient_next + i, elements - i);
			numbers::Number<T> out(output + i, elements - i);
			numbers::Number<T> tmp = _alpha * activation.backward(grad, out);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(gradient_prev + i, elements - i);
			tmp.store(gradient_prev + i, elements - i);

//			T tmp = alpha * activation.backward(gradient_next[i], output[i]);
//			if (beta != zero<U>())
//				tmp += beta * gradient_prev[i];
//			gradient_prev[i] = tmp;
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
				kernel_act_backward<ActivationLinear<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SIGMOID:
				kernel_act_backward<ActivationSigmoid<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_TANH:
				kernel_act_backward<ActivationTanh<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_RELU:
				kernel_act_backward<ActivationRelu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_SELU:
				kernel_act_backward<ActivationSelu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_ELU:
				kernel_act_backward<ActivationElu<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta, elements);
				break;
			case AVOCADO_ACTIVATION_EXPONENTIAL:
				kernel_act_backward<ActivationExponential<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
						elements);
				break;
			case AVOCADO_ACTIVATION_SOFTPLUS:
				kernel_act_backward<ActivationSoftplus<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
						elements);
				break;
			case AVOCADO_ACTIVATION_SOFTSIGN:
				kernel_act_backward<ActivationSoftsign<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(gradient_prev, gradient_next, output, alpha, beta,
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
		using namespace BACKEND_NAMESPACE;

		avStatus_t cudaActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			if (not same_device_type(context, xMem, yMem))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;

			const unsigned int elements = getTensor(yDesc).volume();
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return helper_act_forward(stream, getPointer<float16>(xMem), getPointer<float16>(yMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							activation);
				case AVOCADO_DTYPE_BFLOAT16:
					return helper_act_forward(stream, getPointer<bfloat16>(xMem), getPointer<bfloat16>(yMem), getAlphaValue(alpha), getBetaValue(beta),
							elements, activation);
				case AVOCADO_DTYPE_FLOAT32:
					return helper_act_forward(stream, getPointer<float>(xMem), getPointer<float>(yMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							activation);
				case AVOCADO_DTYPE_FLOAT64:
					return helper_act_forward(stream, getPointer<double>(xMem), getPointer<double>(yMem), getAlphaValue<double>(alpha),
							getBetaValue<double>(beta), elements, activation);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

		avStatus_t cudaActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			if (not same_device_type(context, dxMem, dyMem, yMem))
				return AVOCADO_STATUS_DEVICE_TYPE_MISMATCH;

			const unsigned int elements = getTensor(yDesc).volume();
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return helper_act_backward(stream, getPointer<float>(dxMem), getPointer<float>(dyMem), getPointer<float>(yMem), getAlphaValue(alpha),
							getBetaValue(beta), elements, activation);
				case AVOCADO_DTYPE_FLOAT64:
					return helper_act_backward(stream, getPointer<double>(dxMem), getPointer<double>(dyMem), getPointer<double>(yMem),
							getAlphaValue<double>(alpha), getBetaValue<double>(beta), elements, activation);
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

	} /* namespace backend */
} /* namespace avocado */

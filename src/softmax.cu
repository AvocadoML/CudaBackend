/*
 * softmax.cu
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
	/**
	 * In this version a threadblock calculates softmax in shared memory over several lines of input tensor.
	 */
	template<unsigned int elements, typename T, typename U = T>
	__global__ void kernel_softmax_forward_small_last_dim(const T *input, T *output, U alpha, U beta, unsigned int first_dim, unsigned int last_dim)
	{
		assert(blockDim.x * last_dim <= elements);

		__shared__ T input_storage[elements];
		for (unsigned int i = blockIdx.x * blockDim.x * last_dim; i < first_dim * last_dim; i += gridDim.x * blockDim.x * last_dim)
		{
			for (unsigned int j = threadIdx.x; (j < blockDim.x * last_dim) and ((i + j) < first_dim * last_dim); j += blockDim.x)
				input_storage[j] = input[i + j];
			__syncthreads();

			/* First, find the maximum value of the input */
			T max_element = input_storage[(i + threadIdx.x) * last_dim];
			for (unsigned int j = 0; j < last_dim; j++)
				max_element = max(max_element, input_storage[(i + threadIdx.x) * last_dim + j]);

			/* Second, calculate sum of exponents of input data, shifted by their maximum element */
			T sum = zero<T>();
			for (unsigned int j = 0; j < last_dim; j++)
			{
				input_storage[(i + threadIdx.x) * last_dim + j] = exp(input_storage[(i + threadIdx.x) * last_dim + j] - max_element);
				sum += input_storage[(i + threadIdx.x) * last_dim + j];
			}

			if (sum == zero<T>())
			{
				sum = one<T>() / last_dim;
				for (unsigned int j = 0; j < last_dim; j++)
				{
					T tmp = alpha * sum;
					if (beta != zero<U>())
						tmp += beta * output[i * last_dim + j];
					output[i * last_dim + j] = tmp;
				}
			}
			else
			{
				sum = one<T>() / sum;
				for (unsigned int j = 0; j < last_dim; j++)
				{
					T tmp = alpha * input_storage[(i + threadIdx.x) * last_dim + j] * sum;
					if (beta != zero<U>())
						tmp += beta * output[i * last_dim + j];
					input_storage[(i + threadIdx.x) * last_dim + j] = tmp;
				}
			}

			/* Finally, copy entire block into output array */
			__syncthreads();
			for (unsigned int j = threadIdx.x; (j < blockDim.x * last_dim) and ((i + j) < first_dim * last_dim); j += blockDim.x)
				output[i + j] = input_storage[j];
		}
	}
	/**
	 * In this version single threadblock calculates softmax over single line of input data that fits into shared memory.
	 */
	template<unsigned int elements, typename T, typename U = T>
//	__launch_bounds__(256, 8)
	__global__ void kernel_softmax_forward_medium_last_dim(const T *input, T *output, U alpha, U beta, unsigned int first_dim, unsigned int last_dim)
	{
		assert(blockDim.x == 256 && blockDim.y == 1);
		assert(last_dim <= elements);

		numbers::Number<T> _alpha(alpha);
		numbers::Number<T> _beta(beta);

		__shared__ numbers::Number<T> input_storage[elements];
		__shared__ numbers::Number<T> reduction_storage[256];

		for (unsigned int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			/* First, find the maximum value of the input */
			numbers::Number<T> max_element(input + i * last_dim, last_dim);
			for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
			{
				input_storage[j] = numbers::Number<T>(input + i * last_dim + j, last_dim - j);
				max_element = numbers::max(max_element, input_storage[j]);
			}
			reduction_storage[threadIdx.x] = max_element;
			__syncthreads();

			/* Now reduce the storage array into single element */
			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
			{
				if (threadIdx.x < j)
					reduction_storage[threadIdx.x] = numbers::max(reduction_storage[threadIdx.x], reduction_storage[threadIdx.x + j]);
				__syncthreads();
			}
			max_element = reduction_storage[0];

			/* Second, calculate sum of exponents of input data, shifted by their maximum element */
			numbers::Number<T> sum = numbers::zero<T>();
			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
			{
				input_storage[j] = numbers::exp(input_storage[j] - max_element);
				sum += input_storage[j];
			}
			reduction_storage[threadIdx.x] = sum;
			__syncthreads();

			/* Now sum the storage array into single element */
			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
			{
				if (threadIdx.x < j)
					reduction_storage[threadIdx.x] += reduction_storage[threadIdx.x + j];
				__syncthreads();
			}
			sum = reduction_storage[0];

			if (sum == numbers::zero<T>())
			{
				sum = numbers::one<T>() / numbers::Number<T>(static_cast<float>(last_dim));
				for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
				{
					numbers::Number<T> tmp = _alpha * sum;
					if (_beta != numbers::zero<T>())
						tmp += _beta * numbers::Number<T>(output + i * last_dim + j, last_dim - j);
					tmp.store(output + i * last_dim + j, last_dim - j);
				}
			}
			else
			{
				sum = numbers::one<T>() / sum;
				for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
				{
					numbers::Number<T> tmp = _alpha * input_storage[j] * sum;
					if (_beta != numbers::zero<T>())
						tmp += _beta * numbers::Number<T>(output + i * last_dim + j, last_dim - j);
					tmp.store(output + i * last_dim + j, last_dim - j);
				}
			}
		}

//		__shared__ T input_storage[elements];
//		__shared__ T reduction_storage[256];
//
//		for (unsigned int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			/* First, find the maximum value of the input */
//			T max_element = input[i * last_dim];
//			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				input_storage[j] = input[i * last_dim + j];
//				max_element = max(max_element, input_storage[j]);
//			}
//			reduction_storage[threadIdx.x] = max_element;
//			__syncthreads();
//
//			/* Now reduce the storage array into single element */
//			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
//			{
//				if (threadIdx.x < j)
//					reduction_storage[threadIdx.x] = max(reduction_storage[threadIdx.x], reduction_storage[threadIdx.x + j]);
//				__syncthreads();
//			}
//			max_element = reduction_storage[0];
//
//			/* Second, calculate sum of exponents of input data, shifted by their maximum element */
//			T sum = zero<T>();
//			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//			{
//				input_storage[j] = exp(input_storage[j] - max_element);
//				sum += input_storage[j];
//			}
//			reduction_storage[threadIdx.x] = sum;
//			__syncthreads();
//
//			/* Now sum the storage array into single element */
//			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
//			{
//				if (threadIdx.x < j)
//					reduction_storage[threadIdx.x] += reduction_storage[threadIdx.x + j];
//				__syncthreads();
//			}
//			sum = reduction_storage[0];
//
//			if (sum == zero<T>())
//			{
//				sum = one<T>() / last_dim;
//				for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				{
//					T tmp = alpha * sum;
//					if (beta != zero<U>())
//						tmp += beta * output[i * last_dim + j];
//					output[i * last_dim + j] = tmp;
//				}
//			}
//			else
//			{
//				sum = one<T>() / sum;
//				for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				{
//					T tmp = alpha * input_storage[j] * sum;
//					if (beta != zero<U>())
//						tmp += beta * output[i * last_dim + j];
//					output[i * last_dim + j] = tmp;
//				}
//			}
//		}
	}
	/**
	 * In this version single threadblock calculates softmax over single line of input tensor.
	 */
	template<typename T, typename U = T>
//	__launch_bounds__(256, 8)
	__global__ void kernel_softmax_forward_large_last_dim(const T *input, T *output, U alpha, U beta, unsigned int first_dim, unsigned int last_dim)
	{
		assert(blockDim.x == 256 && blockDim.y == 1);

		numbers::Number<T> _alpha(alpha);
		numbers::Number<T> _beta(beta);
		__shared__ numbers::Number<T> storage[256];

		for (unsigned int i = blockIdx.x; i < first_dim; i += gridDim.x)
		{
			/* First, find the maximum value of the input */
			numbers::Number<T> max_element(input + i * last_dim, last_dim);
			for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
				max_element = max(max_element, numbers::Number<T>(input + i * last_dim + j, last_dim - j));
			storage[threadIdx.x] = max_element;
			__syncthreads();

			/* Now reduce the storage array into single element */
			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
			{
				if (threadIdx.x < j)
					storage[threadIdx.x] = max(storage[threadIdx.x], storage[threadIdx.x + j]);
				__syncthreads();
			}
			max_element = storage[0];

			/* Second, calculate sum of exponents of input data, shifted by their maximum element */
			numbers::Number<T> sum = numbers::zero<T>();
			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
				sum += numbers::exp(numbers::Number<T>(input + i * last_dim + j, last_dim - j) - max_element);
			storage[threadIdx.x] = sum;
			__syncthreads();

			/* Now sum the storage array into single element */
			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
			{
				if (threadIdx.x < j)
					storage[threadIdx.x] += storage[threadIdx.x + j];
				__syncthreads();
			}
			sum = storage[0];

			if (sum == numbers::zero<T>())
			{
				sum = numbers::one<T>() / numbers::Number<T>(static_cast<float>(last_dim));
				for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
				{
					numbers::Number<T> tmp = _alpha * sum;
					if (_beta != numbers::zero<T>())
						tmp += _beta * numbers::Number<T>(output + i * last_dim + j, last_dim - j);
					tmp.store(output + i * last_dim + j, last_dim - j);
				}
			}
			else
			{
				sum = numbers::one<T>() / sum;
				for (unsigned int j = numbers::length<T>() * threadIdx.x; j < last_dim; j += numbers::length<T>() * blockDim.x)
				{
					numbers::Number<T> tmp = _alpha * numbers::exp(numbers::Number<T>(input + i * last_dim + j, last_dim - j) - max_element) * sum;
					if (_beta != numbers::zero<T>())
						tmp += _beta * numbers::Number<T>(output + i * last_dim + j, last_dim - j);
					tmp.store(output + i * last_dim + j, last_dim - j);
				}
			}
		}

//		__shared__ T storage[256];
//
//		for (unsigned int i = blockIdx.x; i < first_dim; i += gridDim.x)
//		{
//			/* First, find the maximum value of the input */
//			T max_element = input[i * last_dim];
//			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				max_element = max(max_element, input[i * last_dim + j]);
//			storage[threadIdx.x] = max_element;
//			__syncthreads();
//
//			/* Now reduce the storage array into single element */
//			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
//			{
//				if (threadIdx.x < j)
//					storage[threadIdx.x] = max(storage[threadIdx.x], storage[threadIdx.x + j]);
//				__syncthreads();
//			}
//			max_element = storage[0];
//
//			/* Second, calculate sum of exponents of input data, shifted by their maximum element */
//			T sum = zero<T>();
//			for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				sum += exp(input[i * last_dim + j] - max_element);
//			storage[threadIdx.x] = sum;
//			__syncthreads();
//
//			/* Now sum the storage array into single element */
//			for (unsigned int j = blockDim.x / 2; j >= 1; j /= 2)
//			{
//				if (threadIdx.x < j)
//					storage[threadIdx.x] += storage[threadIdx.x + j];
//				__syncthreads();
//			}
//			sum = storage[0];
//
//			if (sum == zero<T>())
//			{
//				sum = one<T>() / last_dim;
//				for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				{
//					T tmp = alpha * sum;
//					if (beta != zero<U>())
//						tmp += beta * output[i * last_dim + j];
//					output[i * last_dim + j] = tmp;
//				}
//			}
//			else
//			{
//				sum = one<T>() / sum;
//				for (unsigned int j = threadIdx.x; j < last_dim; j += blockDim.x)
//				{
//					T tmp = alpha * exp(input[i * last_dim + j] - max_element) * sum;
//					if (beta != zero<U>())
//						tmp += beta * output[i * last_dim + j];
//					output[i * last_dim + j] = tmp;
//				}
//			}
//		}
	}

	template<typename T, typename U = T>
	void helper_softmax_forward(cudaStream_t stream, const T *input, T *output, U alpha, U beta, unsigned int first_dim, unsigned int last_dim)
	{
// FIXME
//		if (last_dim <= 4)
//		{
//			dim3 blockDim(256);
//			dim3 gridDim = gridSize<512>(first_dim * last_dim, blockDim.x);
//			kernel_softmax_forward_small_last_dim<1024, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, first_dim, last_dim);
//			return;
//		}
//		if (4 < last_dim and last_dim <= 32)
//		{
//			dim3 blockDim(32);
//			dim3 gridDim = gridSize<1024>(first_dim * last_dim, blockDim.x);
//			kernel_softmax_forward_small_last_dim<1024, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, first_dim, last_dim);
//			return;
//		}
		if (last_dim <= 1024)
//		if (32 < last_dim and last_dim <= 1024)
		{
			dim3 blockDim(256);
			dim3 gridDim = gridSize<1024>(first_dim, 1);
			kernel_softmax_forward_medium_last_dim<1024, T, U> <<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, first_dim, last_dim);
			return;
		}
		if (last_dim > 1024)
		{
			dim3 blockDim(256);
			dim3 gridDim = gridSize<1024>(first_dim, 1);
			kernel_softmax_forward_large_last_dim<<<gridDim, blockDim, 0, stream>>>(input, output, alpha, beta, first_dim, last_dim);
			return;
		}
	}

	template<typename T, typename U = T>
	__global__ void kernel_softmax_backward(T *gradient_prev, const T *gradient_next, const T *output, U alpha, U beta, unsigned int elements)
	{
		ActivationSigmoid<T> activation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			T tmp = alpha * activation.backward(gradient_next[i], output[i]);
			if (beta != zero<U>())
				tmp += beta * gradient_prev[i];
			gradient_prev[i] = tmp;
		}
	}
}

namespace avocado
{
	namespace backend
	{

		avStatus_t cudaSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			unsigned int first_dim, last_dim;
			if (mode == AVOCADO_SOFTMAX_MODE_CHANNEL)
			{
				first_dim = cuda::getTensor(xDesc).volumeWithoutLastDim();
				last_dim = cuda::getTensor(xDesc).lastDim();
			}
			else
			{
				first_dim = cuda::getTensor(xDesc).firstDim();
				last_dim = cuda::getTensor(xDesc).volumeWithoutFirstDim();
			}

			dim3 blockDim(256);
			dim3 gridDim = gridSize<512>(first_dim, blockDim.x);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					helper_softmax_forward(stream, cuda::getPointer<half>(xMem), cuda::getPointer<half>(yMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					helper_softmax_forward(stream, cuda::getPointer<bfloat16>(xMem), cuda::getPointer<bfloat16>(yMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					helper_softmax_forward(stream, cuda::getPointer<float>(xMem), cuda::getPointer<float>(yMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), first_dim, last_dim);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					helper_softmax_forward(stream, cuda::getPointer<double>(xMem), cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha),
							cuda::getBetaValue<double>(beta), first_dim, last_dim);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
		avStatus_t cudaSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha, const avTensorDescriptor_t yDesc,
				const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			const unsigned int elements = cuda::getTensor(yDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim = gridSize<512>(elements, blockDim.x);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_softmax_backward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(dxMem), cuda::getPointer<float>(dyMem),
							cuda::getPointer<float>(yMem), cuda::getAlphaValue(alpha), cuda::getBetaValue(beta), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_softmax_backward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(dxMem), cuda::getPointer<double>(dyMem),
							cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha), cuda::getBetaValue<double>(beta), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

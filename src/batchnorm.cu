/*
 * batchnorm.cu
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

	template<typename T, typename U = T>
	__global__ void kernel_affine_forward(const T *weight, const T *bias, const T *input, T *output, U alpha, U beta, unsigned int first_dim,
			unsigned int last_dim, avActivationType_t activation)
	{
		assert(gridDim.x * blockDim.x <= last_dim);
		assert(blockDim.x == 256 && blockDim.y == 1);
		Store<T, U> store;
		for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < last_dim; j += gridDim.x * blockDim.x)
		{
			T scale = (weight == nullptr) ? one<T>() : weight[j];
			T shift = (bias == nullptr) ? zero<T>() : bias[j];
			for (unsigned int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				U tmp = alpha * activation_forward(activation, input[i * last_dim + j] * scale + shift);
				if (beta != zero<U>())
					tmp += beta * output[i * last_dim + j];
				output[i * last_dim + j] = store(tmp);
			}
		}
	}

	template<typename T>
	__global__ void kernel_batchnorm_forward(const T *mean, const T *variance, const T *scale, const T *shift, const T *input, T *output, T alpha, T beta,
			unsigned int first_dim, unsigned int last_dim, avActivationType_t activation, T epsilon)
	{
		assert(gridDim.x * blockDim.x <= last_dim);
		assert(blockDim.x == 256 && blockDim.y == 1);

		for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < last_dim; j += gridDim.x * blockDim.x)
		{
			T _scale = scale[j] / sqrt(variance[j] + epsilon);
			T _shift = shift[j] - mean[j] * _scale;
			for (unsigned int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				T tmp = alpha * activation_forward(activation, input[i * last_dim + j] * _scale + _shift);
				if (beta != zero<T>())
					tmp += beta * output[i * last_dim + j];
				output[i * last_dim + j] = tmp;
			}
		}
	}

	template<typename T>
	__device__ void block_reduce(T *workspace) noexcept
	{
		for (unsigned int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
				workspace[threadIdx.y * 32 + threadIdx.x] += workspace[(i + threadIdx.y) * 32 + threadIdx.x];
			__syncthreads();
		}
	}
	template<typename T>
	__global__ void kernel_reduce_variance_1(T *workspace, const T* input, const T* mean, unsigned int first_dim, unsigned int last_dim)
	{
		__shared__ T storage[32 * 32];
		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;
			T acc = zero<T>();
			if (idx < last_dim)
			{
				T avg = mean[idx];
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
					acc += square(input[i * last_dim + idx] - avg);
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_reduce(storage);
			if (threadIdx.y == 0 and idx < last_dim)
				workspace[blockIdx.y * last_dim + idx] = storage[0 * 32 + threadIdx.x];
		}
	}
	template<typename T>
	__global__ void kernel_reduce_variance_2(T *variance, const T* workspace, unsigned int first_dim, unsigned int last_dim)
	{
		__shared__ T storage[32 * 32];
		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			T acc = zero<T>();
			if (idx < last_dim)
			{
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
					acc += workspace[i * last_dim + idx];
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_reduce(storage);
			if (threadIdx.y == 0 and idx < last_dim)
				variance[blockIdx.y * last_dim + idx] = storage[0 * 32 + threadIdx.x] / first_dim;
		}
	}

	template<typename T>
	__global__ void kernel_batchnorm_backward_delta_1(T *workspace, const T *input, const T *output, T *gradient_next, const T* mean, const T *variance,
			unsigned int first_dim, unsigned int last_dim, avActivationType_t activation, T epsilon)
	{
		__shared__ T d_sigma[32 * 32];
		__shared__ T d_mu[32 * 32];

		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			T acc_sigma = zero<T>();
			T acc_mu = zero<T>();
			if (idx < last_dim)
			{
				T avg = mean[idx];
				T inv_stddev = one<T>() / sqrt(variance[idx] + epsilon);
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					T tmp_grad = activation_backward(activation, gradient_next[i * last_dim + idx], output[i * last_dim + idx]);
					gradient_next[i * last_dim + idx] = tmp_grad;
					acc_sigma += tmp_grad * (input[i * last_dim + idx] - avg) * inv_stddev;
					acc_mu += tmp_grad;
				}
			}
			d_sigma[threadIdx.y * 32 + threadIdx.x] = acc_sigma;
			d_mu[threadIdx.y * 32 + threadIdx.x] = acc_mu;
			__syncthreads();

			block_reduce(d_sigma);
			block_reduce(d_mu);

			if (threadIdx.y == 0 and idx < last_dim)
			{
				workspace[2 * blockIdx.y * last_dim + idx] = d_sigma[threadIdx.x];
				workspace[(2 * blockIdx.y + 1) * last_dim + idx] = d_mu[threadIdx.x];
			}
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward_delta_2(T* scaleUpdate, T* biasUpdate, T alpha, T beta, T *workspace, unsigned int first_dim,
			unsigned int last_dim)
	{
		__shared__ T d_sigma[32 * 32];
		__shared__ T d_mu[32 * 32];
		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			T acc_sigma = zero<T>();
			T acc_mu = zero<T>();
			if (idx < last_dim)
			{
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					acc_sigma += workspace[2 * i * last_dim + idx];
					acc_mu += workspace[(2 * i + 1) * last_dim + idx];
				}
			}
			d_sigma[threadIdx.y * 32 + threadIdx.x] = acc_sigma;
			d_mu[threadIdx.y * 32 + threadIdx.x] = acc_mu;

			__syncthreads();
			block_reduce(d_sigma);
			block_reduce(d_mu);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				workspace[idx] = d_sigma[threadIdx.x];
				workspace[last_dim + idx] = d_mu[threadIdx.x];

				T tmp_sigma = alpha * d_sigma[threadIdx.x];
				T tmp_mu = alpha * d_mu[threadIdx.x];
				if (beta == zero<T>())
				{
					tmp_sigma += beta * scaleUpdate[idx];
					tmp_mu += beta * biasUpdate[idx];
				}
				scaleUpdate[idx] = tmp_sigma;
				biasUpdate[idx] = tmp_mu;
			}
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward(const T *workspace, const T* mean, const T* variance, const T* scale, const T *input, T *gradient_prev,
			const T *gradient_next, T alpha, T beta, unsigned int first_dim, unsigned int last_dim, T epsilon)
	{
		assert(gridDim.x * blockDim.x <= last_dim);
		assert(blockDim.x == 256 && blockDim.y == 1);

		for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < last_dim; j += gridDim.x * blockDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			T avg = mean[idx];
			T inv_stddev = one<T>() / sqrt(variance[idx] + epsilon);
			T gamma = scale[idx];
			T d_sigma = workspace[idx];
			T d_mu = workspace[last_dim + idx];

			d_sigma = -gamma * inv_stddev * d_sigma / first_dim;
			d_mu = -gamma * inv_stddev * d_mu / first_dim;

			for (unsigned int i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				T tmp = gamma * inv_stddev * gradient_next[i * last_dim + idx] + d_sigma * (input[i * last_dim + idx] - avg) * inv_stddev + d_mu;
				if (beta != zero<T>())
					tmp += beta * gradient_prev[i * last_dim + j];
				gradient_prev[i * last_dim + j] = tmp;
			}
		}
	}

	template<typename T>
	void helper_batchnorm_backward(cuda::ContextDescriptor &context, avActivationType_t activation, T alpha, const cuda::TensorDescriptor& xDesc, const T* xMem,
			const cuda::TensorDescriptor& yDesc, const T* yMem, T beta, const cuda::TensorDescriptor& dxDesc, T* dxMem, const cuda::TensorDescriptor& dyDesc,
			T* dyMem, const cuda::TensorDescriptor& scaleMeanVarDesc, const T* scaleMem, const T* meanMem, const T* varianceMem, T alpha2, T beta2,
			T* scaleUpdateMem, T* biasUpdateMem, double epsilon)
	{
		cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(xDesc, scaleMeanVarDesc);
		cudaStream_t stream = context.getStream();
		dim3 blockDim(32, 32);
		dim3 gridDim1(gridSize<32>(dimensions.last, blockDim.x), gridSize<128>(dimensions.first, blockDim.y), 1);
		dim3 gridDim2(gridDim1.x, 1, 1);

		T* workspace = context.getWorkspace().data<T>();

		kernel_batchnorm_backward_delta_1<<<gridDim1, blockDim, 0, stream>>>(workspace, xMem, yMem, dyMem, meanMem, varianceMem, dimensions.first,
				dimensions.last, activation, static_cast<T>(epsilon));
		kernel_batchnorm_backward_delta_2<<<gridDim2, blockDim, 0, stream>>>(scaleUpdateMem, biasUpdateMem, alpha2, beta2, workspace, dimensions.first,
				dimensions.last);

		dim3 blockDim3(256);
		dim3 gridDim3(gridSize<32>(dimensions.last, blockDim.x), gridSize<1024>(dimensions.first, blockDim.y), 1);
		kernel_batchnorm_backward<<<gridDim3, blockDim3, 0, stream>>>(workspace, meanMem, varianceMem, scaleMem, xMem, dxMem, dyMem, alpha, beta,
				dimensions.first, dimensions.last, static_cast<T>(epsilon));
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(cuda::getTensor(xDesc), cuda::getTensor(bDesc));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			dim3 blockDim(256);
			dim3 gridDim(gridSize<32>(dimensions.last, blockDim.x), gridSize<512>(dimensions.first, blockDim.y), 1);
			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_affine_forward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(wMem), cuda::getPointer<float>(bMem),
							cuda::getPointer<float>(xMem), cuda::getPointer<float>(yMem), cuda::getAlphaValue(alpha), cuda::getBetaValue(beta),
							dimensions.first, dimensions.last, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_affine_forward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(wMem), cuda::getPointer<double>(bMem),
							cuda::getPointer<double>(xMem), cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha),
							cuda::getBetaValue<double>(beta), dimensions.first, dimensions.last, activation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(cuda::getTensor(xDesc), cuda::getTensor(scaleBiasMeanVarDesc));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			dim3 blockDim(256);
			dim3 gridDim(gridSize<32>(dimensions.last, blockDim.x), gridSize<512>(dimensions.first, blockDim.y), 1);
			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_forward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(meanMem), cuda::getPointer<float>(varianceMem),
							cuda::getPointer<float>(scaleMem), cuda::getPointer<float>(biasMem), cuda::getPointer<float>(xMem), cuda::getPointer<float>(yMem),
							cuda::getAlphaValue(alpha), cuda::getBetaValue(beta), dimensions.first, dimensions.last, activation, static_cast<float>(epsilon));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_forward<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(meanMem), cuda::getPointer<double>(varianceMem),
							cuda::getPointer<double>(scaleMem), cuda::getPointer<double>(biasMem), cuda::getPointer<double>(xMem),
							cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha), cuda::getBetaValue<double>(beta), dimensions.first,
							dimensions.last, activation, epsilon);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
				avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
		{
			avStatus_t status = cudaReduceTensor(context, AVOCADO_REDUCE_AVG, nullptr, xDesc, xMem, nullptr, scaleBiasMeanVarDesc, meanMem);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(cuda::getTensor(xDesc), cuda::getTensor(scaleBiasMeanVarDesc));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			dim3 blockDim(32, 32);
			dim3 gridDim1(gridSize<8>(dimensions.last, blockDim.x), gridSize<128>(dimensions.first, blockDim.y), 1);
			dim3 gridDim2(gridDim1.x, 1, 1);

			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					float * workspace = cuda::getContext(context).getWorkspace().data<float>();
					kernel_reduce_variance_1<<<gridDim1, blockDim, 0, stream>>>(workspace, cuda::getPointer<float>(xMem), cuda::getPointer<float>(meanMem),
							dimensions.first, dimensions.last);
					kernel_reduce_variance_2<<<gridDim2, blockDim, 0, stream>>>(cuda::getPointer<float>(varianceMem), workspace, dimensions.first,
							dimensions.last);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double * workspace = cuda::getContext(context).getWorkspace().data<double>();
					kernel_reduce_variance_1<<<gridDim1, blockDim, 0, stream>>>(workspace, cuda::getPointer<double>(xMem), cuda::getPointer<double>(meanMem),
							dimensions.first, dimensions.last);
					kernel_reduce_variance_2<<<gridDim2, blockDim, 0, stream>>>(cuda::getPointer<double>(varianceMem), workspace, dimensions.first,
							dimensions.last);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}

			status = cudaBatchNormInference(context, activation, alpha, xDesc, xMem, beta, yDesc, yMem, scaleBiasMeanVarDesc, scaleMem, biasMem, meanMem,
					varianceMem, epsilon);
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc, avMemoryDescriptor_t dyMem,
				const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t meanMem,
				const avMemoryDescriptor_t varianceMem, const void *alpha2, const void *beta2, avMemoryDescriptor_t scaleUpdateMem,
				avMemoryDescriptor_t biasUpdateMem, double epsilon)
		{
			cuda::getContext(context).setDevice();
			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					helper_batchnorm_backward(cuda::getContext(context), activation, cuda::getAlphaValue(alpha), cuda::getTensor(xDesc),
							cuda::getPointer<float>(xMem), cuda::getTensor(yDesc), cuda::getPointer<float>(yMem), cuda::getBetaValue(beta),
							cuda::getTensor(dxDesc), cuda::getPointer<float>(dxMem), cuda::getTensor(dyDesc), cuda::getPointer<float>(dyMem),
							cuda::getTensor(scaleMeanVarDesc), cuda::getPointer<float>(scaleMem), cuda::getPointer<float>(meanMem),
							cuda::getPointer<float>(varianceMem), cuda::getAlphaValue(alpha2), cuda::getBetaValue(beta2),
							cuda::getPointer<float>(scaleUpdateMem), cuda::getPointer<float>(biasUpdateMem), epsilon);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					helper_batchnorm_backward(cuda::getContext(context), activation, cuda::getAlphaValue<double>(alpha), cuda::getTensor(xDesc),
							cuda::getPointer<double>(xMem), cuda::getTensor(yDesc), cuda::getPointer<double>(yMem), cuda::getBetaValue<double>(beta),
							cuda::getTensor(dxDesc), cuda::getPointer<double>(dxMem), cuda::getTensor(dyDesc), cuda::getPointer<double>(dyMem),
							cuda::getTensor(scaleMeanVarDesc), cuda::getPointer<double>(scaleMem), cuda::getPointer<double>(meanMem),
							cuda::getPointer<double>(varianceMem), cuda::getAlphaValue<double>(alpha2), cuda::getBetaValue<double>(beta2),
							cuda::getPointer<double>(scaleUpdateMem), cuda::getPointer<double>(biasUpdateMem), epsilon);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

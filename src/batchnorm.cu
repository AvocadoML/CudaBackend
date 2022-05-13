/*
 * batchnorm.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"
#include "reduction_utils.cuh"
#include "numbers/numbers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;
	using namespace numbers;

	template<typename T, typename U = T>
	__global__ void kernel_affine_forward(const T *weight, const T *bias, const T *input, T *output, U alpha, U beta, uint32_t first_dim, uint32_t last_dim,
			avActivationType_t activation)
	{
		assert(gridDim.x * blockDim.x <= last_dim);
		assert(blockDim.x == 256 && blockDim.y == 1);
		Number<T> _alpha(alpha);
		Number<T> _beta(beta);
		for (uint32_t j = length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); j < last_dim; j += length<T>() * gridDim.x * blockDim.x)
		{
			Number<T> scale(weight + j, last_dim - j);
			Number<T> shift(bias + j, last_dim - j);
			for (uint32_t i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				Number<T> src(input + i * last_dim + j, last_dim - j);
				Number<T> tmp = _alpha * activation_forward(activation, src * scale + shift);
				if (_beta != numbers::zero<T>())
					tmp += _beta * Number<T>(output + i * last_dim + j, last_dim - j);
				tmp.store(output + i * last_dim + j, last_dim - j);
			}
		}
	}

	template<typename T>
	__global__ void kernel_batchnorm_forward(const T *mean, const T *variance, const T *scale, const T *shift, const T *input, T *output, T alpha, T beta,
			uint32_t first_dim, uint32_t last_dim, avActivationType_t activation, T epsilon)
	{
		Number<T> _alpha(alpha);
		Number<T> _beta(beta);
		for (uint32_t j = length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); j < last_dim; j += length<T>() * gridDim.x * blockDim.x)
		{
			Number<T> _scale = Number<T>(scale + j, last_dim - j) / numbers::sqrt(Number<T>(variance + j, last_dim - j) + Number<T>(epsilon));
			Number<T> _shift = Number<T>(shift + j, last_dim - j) - Number<T>(mean + j, last_dim - j) * _scale;
			for (uint32_t i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				Number<T> src(input + i * last_dim + j, last_dim - j);
				Number<T> tmp = _alpha * activation_forward(activation, src * _scale + _shift);
				if (_beta != numbers::zero<T>())
					tmp += _beta * Number<T>(output + i * last_dim + j, last_dim - j);
				tmp.store(output + i * last_dim + j, last_dim - j);
			}
		}
	}

	template<typename T>
	__global__ void kernel_reduce_variance_1(T *workspace, const T* input, const T* mean, uint32_t first_dim, uint32_t last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ ReduceAdd<T> storage[32 * 32];
		for (uint32_t j = length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += length<T>() * blockDim.x * gridDim.x)
		{
			uint32_t idx = j + length<T>() * threadIdx.x;
			ReduceAdd<T> acc;
			acc.init();
			if (idx < last_dim)
			{
				Number<T> avg(mean + idx, last_dim - idx);
				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					Number<T> src(input + i * last_dim + idx, last_dim - idx);
					acc.accumulate(square(src - avg));
				}
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_broadcasted_reduction(storage);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				Number<T> tmp = (Number<T> ) storage[0 * 32 + threadIdx.x];
				tmp.store(workspace + blockIdx.y * last_dim + idx, last_dim - idx);
			}
		}
	}
	template<typename T>
	__global__ void kernel_reduce_variance_2(T *variance, const T* workspace, uint32_t first_dim, uint32_t last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ ReduceAdd<T> storage[32 * 32];
		for (uint32_t j = numbers::length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			uint32_t idx = j + numbers::length<T>() * threadIdx.x;

			ReduceAdd<T> acc;
			acc.init();
			if (idx < last_dim)
			{
				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					Number<T> tmp = Number<T>(workspace + i * last_dim + idx, last_dim - idx);
					acc.combine_partial(tmp);
				}
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_broadcasted_reduction(storage);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				storage[0 * 32 + threadIdx.x].final_action();
				numbers::Number<T> tmp = (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]) / Number<T>(static_cast<double>(first_dim));
				tmp.store(variance + blockIdx.y * last_dim + idx, last_dim - idx);
			}
		}

//		__shared__ Number<T> storage[32 * 32];
//		for (uint32_t j = length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += length<T>() * blockDim.x * gridDim.x)
//		{
//			uint32_t idx = j + threadIdx.x;
//
//			Number<T> acc = numbers::zero<T>();
//			if (idx < last_dim)
//			{
//				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//					acc += workspace[i * last_dim + idx];
//			}
//			storage[threadIdx.y * 32 + threadIdx.x] = acc;
//
//			__syncthreads();
//			block_broadcasted_reduction(storage);
//			if (threadIdx.y == 0 and idx < last_dim)
//			{
//				Number<T> tmp = storage[0 * 32 + threadIdx.x];
//				tmp.store(variance + blockIdx.y * last_dim + idx, last_dim - idx);
//			}
//		}
	}

	template<typename T>
	__global__ void kernel_batchnorm_backward_delta_1(T *workspace, const T *input, const T *output, T *gradient_next, const T* mean, const T *variance,
			uint32_t first_dim, uint32_t last_dim, avActivationType_t activation, T epsilon)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ ReduceAdd<T> d_sigma[32 * 32];
		__shared__ ReduceAdd<T> d_mu[32 * 32];

		for (uint32_t j = length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += length<T>() * blockDim.x * gridDim.x)
		{
			uint32_t idx = j + length<T>() * threadIdx.x;

			ReduceAdd<T> acc_sigma;
			ReduceAdd<T> acc_mu;
			acc_sigma.init();
			acc_mu.init();
			if (idx < last_dim)
			{
				Number<T> avg(mean + idx, last_dim - idx);
				Number<T> inv_stddev = numbers::one<T>() / numbers::sqrt(Number<T>(variance + idx, last_dim - idx) + Number<T>(epsilon));
				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					Number<T> tmp_grad(gradient_next + i * last_dim + idx, last_dim - idx);
					Number<T> tmp_out(output + i * last_dim + idx, last_dim - idx);
					tmp_grad = activation_backward(activation, tmp_grad, tmp_out);
					tmp_grad.store(gradient_next + i * last_dim + idx, last_dim - idx);

					Number<T> tmp_in(input + i * last_dim + idx, last_dim - idx);
					acc_sigma.accumulate(tmp_grad * (tmp_in - avg) * inv_stddev);
					acc_mu.accumulate(tmp_grad);
				}
			}
			d_sigma[threadIdx.y * 32 + threadIdx.x] = acc_sigma;
			d_mu[threadIdx.y * 32 + threadIdx.x] = acc_mu;

			__syncthreads();
			block_broadcasted_reduction(d_sigma);
			block_broadcasted_reduction(d_mu);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				Number<T> tmp_sigma = (Number<T> ) d_sigma[threadIdx.x];
				Number<T> tmp_mu = (Number<T> ) d_mu[threadIdx.x];
				tmp_sigma.store(workspace + 2 * blockIdx.y * last_dim + idx, last_dim - idx);
				tmp_mu.store(workspace + (2 * blockIdx.y + 1) * last_dim + idx, last_dim - idx);
			}
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward_delta_2(T* scaleUpdate, T* biasUpdate, T alpha, T beta, T *workspace, uint32_t first_dim, uint32_t last_dim)
	{
		assert(blockDim.x == 32);
		assert(blockDim.y == 32);
		__shared__ ReduceAdd<T> d_sigma[32 * 32];
		__shared__ ReduceAdd<T> d_mu[32 * 32];
		for (uint32_t j = length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += length<T>() * blockDim.x * gridDim.x)
		{
			uint32_t idx = j + length<T>() * threadIdx.x;

			ReduceAdd<T> acc_sigma;
			ReduceAdd<T> acc_mu;
			acc_sigma.init();
			acc_mu.init();
			if (idx < last_dim)
			{
				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					Number<T> tmp_sigma(workspace + 2 * i * last_dim + idx, last_dim - idx);
					Number<T> tmp_mu(workspace + (2 * i + 1) * last_dim + idx, last_dim - idx);
					acc_sigma.combine_partial(tmp_sigma);
					acc_mu.combine_partial(tmp_mu);
				}
			}
			d_sigma[threadIdx.y * 32 + threadIdx.x] = acc_sigma;
			d_mu[threadIdx.y * 32 + threadIdx.x] = acc_mu;

			__syncthreads();
			block_broadcasted_reduction(d_sigma);
			block_broadcasted_reduction(d_mu);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				Number<T> tmp_sigma = (Number<T> ) d_sigma[threadIdx.x];
				Number<T> tmp_mu = (Number<T> ) d_mu[threadIdx.x];
				tmp_sigma.store(workspace + idx, last_dim - idx);
				tmp_mu.store(workspace + idx + last_dim, last_dim - idx);

				tmp_sigma *= Number<T>(alpha);
				tmp_mu *= Number<T>(alpha);
				if (beta != scalar_zero<T>())
				{
					tmp_sigma += Number<T>(beta) * Number<T>(scaleUpdate + idx, last_dim - idx);
					tmp_mu += Number<T>(beta) * Number<T>(biasUpdate + idx, last_dim - idx);
				}
				tmp_sigma.store(scaleUpdate + idx, last_dim - idx);
				tmp_mu.store(biasUpdate + idx, last_dim - idx);
			}
		}
	}
	template<typename T>
	__global__ void kernel_batchnorm_backward(const T *workspace, const T* mean, const T* variance, const T* scale, const T *input, T *gradient_prev,
			const T *gradient_next, T alpha, T beta, uint32_t first_dim, uint32_t last_dim, T epsilon)
	{
		assert(blockDim.x == 256 && blockDim.y == 1);

		for (uint32_t j = blockIdx.x * blockDim.x + threadIdx.x; j < last_dim; j += gridDim.x * blockDim.x)
		{
			T avg = mean[j];
			T inv_stddev = scalar_one<T>() / sqrt(variance[j] + epsilon);
			T gamma = scale[j];
			T d_sigma = workspace[j];
			T d_mu = workspace[last_dim + j];

			d_sigma = -gamma * inv_stddev * d_sigma / first_dim;
			d_mu = -gamma * inv_stddev * d_mu / first_dim;

			for (uint32_t i = blockIdx.y; i < first_dim; i += gridDim.y)
			{
				T tmp = gamma * inv_stddev * gradient_next[i * last_dim + j] + d_sigma * (input[i * last_dim + j] - avg) * inv_stddev + d_mu;
				tmp *= alpha;
				if (beta != scalar_zero<T>())
					tmp += beta * gradient_prev[i * last_dim + j];
				gradient_prev[i * last_dim + j] = tmp;
			}
		}
	}

	template<typename T>
	void helper_batchnorm_backward(ContextDescriptor &context, avActivationType_t activation, T alpha, const TensorDescriptor& xDesc, const T* xMem,
			const TensorDescriptor& yDesc, const T* yMem, T beta, const TensorDescriptor& dxDesc, T* dxMem, const TensorDescriptor& dyDesc, T* dyMem,
			const TensorDescriptor& scaleMeanVarDesc, const T* scaleMem, const T* meanMem, const T* varianceMem, T alpha2, T beta2, T* scaleUpdateMem,
			T* biasUpdateMem, double epsilon)
	{
		BroadcastedDimensions dimensions = getBroadcastDimensions(xDesc, scaleMeanVarDesc);
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
		using namespace BACKEND_NAMESPACE;

		avStatus_t cudaAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(bDesc));
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			dim3 blockDim(256);
			dim3 gridDim(gridSize<32>(dimensions.last, blockDim.x), gridSize<512>(dimensions.first, blockDim.y), 1);
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_affine_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(wMem), getPointer<float>(bMem), getPointer<float>(xMem),
							getPointer<float>(yMem), getAlphaValue(alpha), getBetaValue(beta), dimensions.first, dimensions.last, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_affine_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(wMem), getPointer<double>(bMem), getPointer<double>(xMem),
							getPointer<double>(yMem), getAlphaValue<double>(alpha), getBetaValue<double>(beta), dimensions.first, dimensions.last, activation);
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
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleBiasMeanVarDesc));
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			dim3 blockDim(256);
			dim3 gridDim(gridSize<32>(dimensions.last, blockDim.x), gridSize<512>(dimensions.first, blockDim.y), 1);
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_batchnorm_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(meanMem), getPointer<float>(varianceMem),
							getPointer<float>(scaleMem), getPointer<float>(biasMem), getPointer<float>(xMem), getPointer<float>(yMem), getAlphaValue(alpha),
							getBetaValue(beta), dimensions.first, dimensions.last, activation, static_cast<float>(epsilon));
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_batchnorm_forward<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(meanMem), getPointer<double>(varianceMem),
							getPointer<double>(scaleMem), getPointer<double>(biasMem), getPointer<double>(xMem), getPointer<double>(yMem),
							getAlphaValue<double>(alpha), getBetaValue<double>(beta), dimensions.first, dimensions.last, activation, epsilon);
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

			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(xDesc), getTensor(scaleBiasMeanVarDesc));
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			dim3 blockDim(32, 32);
			dim3 gridDim1(gridSize<8>(dimensions.last, blockDim.x), gridSize<128>(dimensions.first, blockDim.y), 1);
			dim3 gridDim2(gridDim1.x, 1, 1);

			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					float * workspace = getContext(context).getWorkspace().data<float>();
					kernel_reduce_variance_1<<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<float>(xMem), getPointer<float>(meanMem),
							dimensions.first, dimensions.last);
					kernel_reduce_variance_2<<<gridDim2, blockDim, 0, stream>>>(getPointer<float>(varianceMem), workspace, dimensions.first, dimensions.last);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double * workspace = getContext(context).getWorkspace().data<double>();
					kernel_reduce_variance_1<<<gridDim1, blockDim, 0, stream>>>(workspace, getPointer<double>(xMem), getPointer<double>(meanMem),
							dimensions.first, dimensions.last);
					kernel_reduce_variance_2<<<gridDim2, blockDim, 0, stream>>>(getPointer<double>(varianceMem), workspace, dimensions.first, dimensions.last);
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
			getContext(context).setDevice();
			switch (getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					helper_batchnorm_backward(getContext(context), activation, getAlphaValue(alpha), getTensor(xDesc), getPointer<float>(xMem),
							getTensor(yDesc), getPointer<float>(yMem), getBetaValue(beta), getTensor(dxDesc), getPointer<float>(dxMem), getTensor(dyDesc),
							getPointer<float>(dyMem), getTensor(scaleMeanVarDesc), getPointer<float>(scaleMem), getPointer<float>(meanMem),
							getPointer<float>(varianceMem), getAlphaValue(alpha2), getBetaValue(beta2), getPointer<float>(scaleUpdateMem),
							getPointer<float>(biasUpdateMem), epsilon);
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					helper_batchnorm_backward(getContext(context), activation, getAlphaValue<double>(alpha), getTensor(xDesc), getPointer<double>(xMem),
							getTensor(yDesc), getPointer<double>(yMem), getBetaValue<double>(beta), getTensor(dxDesc), getPointer<double>(dxMem),
							getTensor(dyDesc), getPointer<double>(dyMem), getTensor(scaleMeanVarDesc), getPointer<double>(scaleMem),
							getPointer<double>(meanMem), getPointer<double>(varianceMem), getAlphaValue<double>(alpha2), getBetaValue<double>(beta2),
							getPointer<double>(scaleUpdateMem), getPointer<double>(biasUpdateMem), epsilon);
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

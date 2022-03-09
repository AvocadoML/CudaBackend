/*
 * convert.cu
 *
 *  Created on: Sep 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "utilities.hpp"
#include "activations.cuh"
#include "reduction_utils.cuh"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;

//	template<class Accumulator>
//	__device__ void block_reduce_linear(Accumulator *ptr) noexcept
//	{
////		assert(ispow2(blockDim.x));
////		for (uint32_t i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
////		{
////			if (threadIdx.x < i)
////				ptr[threadIdx.x].combine_partial(ptr[threadIdx.x + i]);
////			__syncthreads();
////		}
//	}
//	template<class Accumulator, typename T>
//	__global__ void kernel_reduce_linear_1(T *dst, const T* src, uint32_t elements)
//	{
////		__shared__ Accumulator storage[1024];
////
////		Accumulator acc;
////		for (uint32_t i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
////			acc.accumulate(src[i]);
////		storage[threadIdx.x] = acc;
////
////		__syncthreads();
////		block_reduce_linear(storage);
////		if (threadIdx.x == 0)
////			reinterpret_cast<Accumulator*>(dst)[blockIdx.x] = storage[0];
//	}
//	template<class Accumulator, typename T, typename U = T>
//	__global__ void kernel_reduce_linear_2(T *dst, const T* src, U alpha, U beta)
//	{
////		assert(blockDim.x <= 1024);
////		__shared__ Accumulator storage[1024];
////		storage[threadIdx.x] = reinterpret_cast<const Accumulator*>(src)[threadIdx.x];
////		__syncthreads();
////		block_reduce_linear(storage);
////
////		if (threadIdx.x == 0)
////		{
////			numbers::Number<T> _alpha(alpha);
////			numbers::Number<T> _beta(beta);
////			storage[0].final_action();
////			numbers::Number<T> tmp = _alpha * (numbers::Number<T>) (storage[0]);
////			if (_beta != numbers::zero<T>())
////				tmp += _beta * numbers::Number<T>(dst, 1);
////			tmp.store(dst, 1);
////		}
//	}
//
//	template<class Accumulator, typename T>
//	__device__ void block_reduce_broadcasted(numbers::Number<T> *ptr) noexcept
//	{
//		assert(blockDim.x == 32);
//		assert(blockDim.y == 32);
//		for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
//		{
//			if (threadIdx.y < i)
//			{
//				int x = threadIdx.y * 32 + threadIdx.x;
//				int y = (i + threadIdx.y) * 32 + threadIdx.x;
////				float v0 = (numbers::Number<float>) (ptr[x]);
////				float v1 = (numbers::Number<float>) (ptr[y]);
////				printf("to %i (%f) add from %i (%f)\n", x, v0, y, v1);
////				ptr[threadIdx.y * 32 + threadIdx.x].combine_partial(ptr[(i + threadIdx.y) * 32 + threadIdx.x]);
//			}
//			__syncthreads();
//		}
//	}
//	template<class Accumulator, typename T>
//	__global__ void kernel_reduce_broadcasted_1(T *dst, const T* src, uint32_t first_dim, uint32_t last_dim)
//	{
//		__shared__ T storage[32 * 32];
////		Accumulator accumulator;
//		for (uint32_t j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
//		{
//			uint32_t idx = j + threadIdx.x;
//
//			T acc; //= accumulator.init();
////			if (idx < last_dim)
////			{
////				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
////				{
////					printf("thread %i,%i x %i,%i loading %i,%i\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, idx);
////					numbers::Number<T> tmp(src + i * last_dim + idx, last_dim - idx);
////					acc.accumulate(tmp);
////				}
////			}
//			assert((threadIdx.y * 32 + threadIdx.x) < 1024);
//			storage[threadIdx.y * 32 + threadIdx.x] = (T) threadIdx.x; //numbers::Number<T>((float) (threadIdx.x)); //acc;
//
//			__syncthreads();
////			block_reduce_broadcasted(storage);
//			if (threadIdx.y == 0 and idx < last_dim)
//			{
////				numbers::Number<T> tmp = numbers::one<T>();
////				numbers::Number<T> tmp = storage[0 * 32 + threadIdx.x];
//				T tmp = storage[0 * 32 + threadIdx.x];
//				assert((blockIdx.y * last_dim + idx) < 3 * 29);
//				dst[blockIdx.y * last_dim + idx] = tmp;
////				tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
//
////				tmp.store(dst, 1);
//			}
//			__syncthreads();
//		}
//
////		__shared__ Accumulator storage[32 * 32];
////		for (uint32_t j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
////		{
////			uint32_t idx = j + threadIdx.x;
////
////			Accumulator acc;
//////			if (idx < last_dim)
//////			{
//////				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//////				{
//////					printf("thread %i,%i x %i,%i loading %i,%i\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, idx);
//////					numbers::Number<T> tmp(src + i * last_dim + idx, last_dim - idx);
//////					acc.accumulate(tmp);
//////				}
//////			}
////			storage[threadIdx.y * 32 + threadIdx.x] = numbers::Number<T>((float) (threadIdx.x)); //acc;
////
////			__syncthreads();
//////			block_reduce_broadcasted(storage);
////			if (threadIdx.y == 0 and idx < last_dim)
////			{
//////				numbers::Number<T> tmp = numbers::one<T>();
////				numbers::Number<T> tmp = (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]);
////				tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
////
//////				tmp.store(dst, 1);
////			}
////		}
//	}
//	template<class Accumulator, typename T, typename U = T>
//	__global__ void kernel_reduce_broadcasted_2(T *dst, const T* src, U alpha, U beta, uint32_t first_dim, uint32_t last_dim)
//	{
//		__shared__ Accumulator storage[32 * 32];
//		for (uint32_t j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
//		{
//			uint32_t idx = j + threadIdx.x;
//
//			Accumulator acc;
//			if (idx < last_dim)
//			{
//				for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//					acc.combine_partial(reinterpret_cast<const Accumulator*>(src)[i * last_dim + idx]);
//			}
//			storage[threadIdx.y * 32 + threadIdx.x] = acc;
//
//			__syncthreads();
//			block_reduce_broadcasted(storage);
//			if (threadIdx.y == 0 and idx < last_dim)
//			{
//				numbers::Number<T> _alpha(alpha);
//				numbers::Number<T> _beta(beta);
//				storage[0 * 32 + threadIdx.x].final_action();
//				numbers::Number<T> tmp = _alpha * (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]);
//				if (_beta != numbers::zero<T>())
//					tmp += _beta * numbers::Number<T>(dst + blockIdx.y * last_dim + idx, last_dim - idx);
//				tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
//			}
//		}
//	}

	template<class Accumulator, typename T, typename U = T>
	avStatus_t helper_reduce_tensor(cudaStream_t stream, T* output, const T *input, const U alpha, const U beta, cuda::BroadcastedDimensions dimensions,
			cuda::MemoryDescriptor &workspace)
	{
		assert(output != nullptr);
		assert(input != nullptr);
		if (dimensions.last == 1) // output is a single element
			return launch_linear_reduction<Accumulator, T, U>(stream, output, input, alpha, beta, dimensions.first, workspace);
		else
			return launch_broadcasted_reduction<Accumulator, T, U>(stream, output, input, alpha, beta, dimensions.first, dimensions.last, workspace);
	}
	template<typename T, typename U = T>
	avStatus_t launcher_reduce_tensor(cudaStream_t stream, T* dst, const T *input, const U alpha, const U beta, cuda::BroadcastedDimensions dimensions,
			avReduceOp_t operation, cuda::MemoryDescriptor &workspace)
	{
		switch (operation)
		{
			case AVOCADO_REDUCE_ADD:
				return helper_reduce_tensor<ReduceAdd<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_MUL:
				return helper_reduce_tensor<ReduceMul<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_MIN:
				return helper_reduce_tensor<ReduceMin<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_MAX:
				return helper_reduce_tensor<ReduceMax<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_AMAX:
				return helper_reduce_tensor<ReduceAMax<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_AVG:
				return helper_reduce_tensor<ReduceAdd<T>, T, U>(stream, dst, input, alpha / dimensions.first, beta, dimensions, workspace);
			case AVOCADO_REDUCE_NORM1:
				return helper_reduce_tensor<ReduceNorm1<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_NORM2:
				return helper_reduce_tensor<ReduceNorm2<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
			case AVOCADO_REDUCE_MUL_NO_ZEROS:
				return helper_reduce_tensor<ReduceMulNoZeros<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
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

		avStatus_t cudaReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(cuda::getTensor(aDesc), cuda::getTensor(cDesc));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(aDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return launcher_reduce_tensor(stream, cuda::getPointer<float16>(cMem), cuda::getPointer<float16>(aMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace());
				case AVOCADO_DTYPE_BFLOAT16:
					return launcher_reduce_tensor(stream, cuda::getPointer<bfloat16>(cMem), cuda::getPointer<bfloat16>(aMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace());
				case AVOCADO_DTYPE_FLOAT32:
					return launcher_reduce_tensor(stream, cuda::getPointer<float>(cMem), cuda::getPointer<float>(aMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace());
				case AVOCADO_DTYPE_FLOAT64:
					return launcher_reduce_tensor(stream, cuda::getPointer<double>(cMem), cuda::getPointer<double>(aMem), cuda::getAlphaValue<double>(alpha),
							cuda::getBetaValue<double>(beta), dimensions, operation, cuda::getContext(context).getWorkspace());
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

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

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	struct limits
	{
		__device__ T max() const noexcept
		{
			return zero<T>();
		}
	};
	template<>
	struct limits<float16>
	{
		__device__ float max() const noexcept
		{
			return 65504;
		}
	};
	template<>
	struct limits<bfloat16>
	{
		__device__ float max() const noexcept
		{
			return 3.4028e+38f;
		}
	};
	template<>
	struct limits<float>
	{
		__device__ float max() const noexcept
		{
			return 3.40282346638528859811704183484516925e+38f;
		}
	};
	template<>
	struct limits<double>
	{
		__device__ double max() const noexcept
		{
			return 1.79769313486231570814527423731704357e+308;
		}
	};

	template<typename T>
	class ReduceAdd
	{
		numbers::Number<T> acc = numbers::zero<T>();
	public:
		__device__ ReduceAdd() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc += x;
		}
		__device__ void combine_partial(ReduceAdd other) noexcept
		{
			acc += other.acc;
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceAdd& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceMul
	{
		numbers::Number<T> acc = numbers::one<T>();
	public:
		__device__ ReduceMul() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc *= x;
		}
		__device__ void combine_partial(ReduceMul other) noexcept
		{
			acc *= other.acc;
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceMul& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceMin
	{
		numbers::Number<T> acc = numbers::Number<T>(limits<T>().max());
	public:
		__device__ ReduceMin() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			this->acc = numbers::min(this->acc, x);
		}
		__device__ void combine_partial(ReduceMin other) noexcept
		{
			this->acc = numbers::min(this->acc, other.acc);
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceMin& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceMax
	{
		numbers::Number<T> acc = numbers::Number<T>(-limits<T>().max());
	public:
		__device__ ReduceMax() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc = numbers::max(acc, x);
		}
		__device__ void combine_partial(ReduceMax other) noexcept
		{
			acc = numbers::max(acc, other.acc);
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceMax& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceAMax
	{
		numbers::Number<T> acc = numbers::zero<T>();
	public:
		__device__ ReduceAMax() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc = numbers::max(acc, numbers::abs(x));
		}
		__device__ void combine_partial(ReduceAMax other) noexcept
		{
			acc = numbers::max(acc, other.acc);
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceAMax& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceNorm1
	{
		numbers::Number<T> acc = numbers::zero<T>();
	public:
		__device__ ReduceNorm1() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc += numbers::abs(x);
		}
		__device__ void combine_partial(ReduceNorm1 other) noexcept
		{
			acc += other.acc;
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceNorm1& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceNorm2
	{
		numbers::Number<T> acc = numbers::zero<T>();
	public:
		__device__ ReduceNorm2() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			acc += numbers::square(x);
		}
		__device__ void combine_partial(ReduceNorm2 other) noexcept
		{
			acc += other.acc;
		}
		__device__ void final_action() noexcept
		{
			acc = numbers::sqrt(acc);
		}
		__device__ ReduceNorm2& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};
	template<typename T>
	class ReduceMulNoZeros
	{
		numbers::Number<T> acc = numbers::one<T>();
	public:
		__device__ ReduceMulNoZeros() = default;
		__device__ void accumulate(numbers::Number<T> x) noexcept
		{
			if (x != numbers::zero<T>())
				acc *= x;
		}
		__device__ void combine_partial(ReduceMulNoZeros other) noexcept
		{
			acc *= other.acc;
		}
		__device__ void final_action() noexcept
		{
		}
		__device__ ReduceMulNoZeros& operator=(numbers::Number<T> value) noexcept
		{
			acc = value;
			return *this;
		}
		__device__ operator numbers::Number<T>() const noexcept
		{
			return acc;
		}
	};

	template<class Accumulator>
	__device__ void block_reduce_linear(Accumulator *ptr) noexcept
	{
		assert(ispow2(blockDim.x));
		for (unsigned int i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.x < i)
				ptr[threadIdx.x].combine_partial(ptr[threadIdx.x + i]);
			__syncthreads();
		}
	}
	template<class Accumulator, typename T>
	__global__ void kernel_reduce_linear_1(T *dst, const T* src, unsigned int elements)
	{
		__shared__ Accumulator storage[1024];

		Accumulator acc;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
			acc.accumulate(src[i]);
		storage[threadIdx.x] = acc;

		__syncthreads();
		block_reduce_linear(storage);
		if (threadIdx.x == 0)
			reinterpret_cast<Accumulator*>(dst)[blockIdx.x] = storage[0];
	}
	template<class Accumulator, typename T, typename U = T>
	__global__ void kernel_reduce_linear_2(T *dst, const T* src, U alpha, U beta)
	{
		__shared__ Accumulator storage[1024];
		storage[threadIdx.x] = reinterpret_cast<const Accumulator*>(src)[threadIdx.x];
		__syncthreads();
		block_reduce_linear(storage);

		if (threadIdx.x == 0)
		{
			numbers::Number<T> _alpha(alpha);
			numbers::Number<T> _beta(beta);
			storage[0].final_action();
			numbers::Number<T> tmp = _alpha * (numbers::Number<T>) (storage[0]);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(dst, 1);
			tmp.store(dst, 1);
		}
	}

	template<class Accumulator>
	__device__ void block_reduce_broadcasted(Accumulator *ptr) noexcept
	{
		for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.y < i)
				ptr[threadIdx.y * 32 + threadIdx.x].combine_partial(ptr[(i + threadIdx.y) * 32 + threadIdx.x]);
			__syncthreads();
		}
	}
	template<class Accumulator, typename T>
	__global__ void kernel_reduce_broadcasted_1(T *dst, const T* src, unsigned int first_dim, unsigned int last_dim)
	{
		__shared__ Accumulator storage[32 * 32];
		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			Accumulator acc;
			if (idx < last_dim)
			{
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
					acc.accumulate(src[i * last_dim + idx]);
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_reduce_broadcasted(storage);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				numbers::Number<T> tmp = (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]);
				tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
			}
		}
	}
	template<class Accumulator, typename T, typename U = T>
	__global__ void kernel_reduce_broadcasted_2(T *dst, const T* src, U alpha, U beta, unsigned int first_dim, unsigned int last_dim)
	{
		__shared__ Accumulator storage[32 * 32];
		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
		{
			unsigned int idx = j + threadIdx.x;

			Accumulator acc;
			if (idx < last_dim)
			{
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
					acc.combine_partial(reinterpret_cast<const Accumulator*>(src)[i * last_dim + idx]);
			}
			storage[threadIdx.y * 32 + threadIdx.x] = acc;

			__syncthreads();
			block_reduce_broadcasted(storage);
			if (threadIdx.y == 0 and idx < last_dim)
			{
				numbers::Number<T> _alpha(alpha);
				numbers::Number<T> _beta(beta);
				storage[0 * 32 + threadIdx.x].final_action();
				numbers::Number<T> tmp = _alpha * (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]);
				if (_beta != numbers::zero<T>())
					tmp += _beta * numbers::Number<T>(dst + blockIdx.y * last_dim + idx, last_dim - idx);
				tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
			}
		}
	}

	template<class Op, typename T, typename U = T>
	void helper_reduce_tensor(cudaStream_t stream, T* output, const T *input, const U alpha, const U beta, cuda::BroadcastedDimensions dimensions, T* workspace)
	{
		assert(output != nullptr);
		assert(input != nullptr);
		assert(workspace != nullptr);
		if (dimensions.last == 1) // output is a single element
		{
			const int partial_results = 64; // must be power of 2
			kernel_reduce_linear_1<Op, T> <<<partial_results, 1024, 0, stream>>>(workspace, input, dimensions.first);
			kernel_reduce_linear_2<Op, T, U> <<<1, partial_results, 0, stream>>>(output, workspace, alpha, beta);
		}
		else
		{
			dim3 blockDim(32, 32);
			dim3 gridDim1(8, 128);
			kernel_reduce_broadcasted_1<Op, T> <<<gridDim1, blockDim, 0, stream>>>(workspace, input, dimensions.first, dimensions.last);

			dim3 gridDim2(8, 1);
			kernel_reduce_broadcasted_2<Op, T, U> <<<gridDim2, blockDim, 0, stream>>>(output, workspace, alpha, beta, dimensions.first, dimensions.last);
		}
	}
	template<typename T, typename U = T>
	avStatus_t launcher_reduce_tensor(cudaStream_t stream, T* dst, const T *input, const U alpha, const U beta, cuda::BroadcastedDimensions dimensions,
			avReduceOp_t operation, T* workspace)
	{
		switch (operation)
		{
			case AVOCADO_REDUCE_ADD:
				helper_reduce_tensor<ReduceAdd<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MUL:
				helper_reduce_tensor<ReduceMul<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MIN:
				helper_reduce_tensor<ReduceMin<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MAX:
				helper_reduce_tensor<ReduceMax<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_AMAX:
				helper_reduce_tensor<ReduceAMax<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_AVG:
				helper_reduce_tensor<ReduceAdd<T>, T, U>(stream, dst, input, alpha / dimensions.first, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_NORM1:
				helper_reduce_tensor<ReduceNorm1<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_NORM2:
				helper_reduce_tensor<ReduceNorm2<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
				break;
			case AVOCADO_REDUCE_MUL_NO_ZEROS:
				helper_reduce_tensor<ReduceMulNoZeros<T>, T, U>(stream, dst, input, alpha, beta, dimensions, workspace);
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
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace().data<float16>());
				case AVOCADO_DTYPE_BFLOAT16:
					return launcher_reduce_tensor(stream, cuda::getPointer<bfloat16>(cMem), cuda::getPointer<bfloat16>(aMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace().data<bfloat16>());
				case AVOCADO_DTYPE_FLOAT32:
					return launcher_reduce_tensor(stream, cuda::getPointer<float>(cMem), cuda::getPointer<float>(aMem), cuda::getAlphaValue(alpha),
							cuda::getBetaValue(beta), dimensions, operation, cuda::getContext(context).getWorkspace().data<float>());
				case AVOCADO_DTYPE_FLOAT64:
					return launcher_reduce_tensor(stream, cuda::getPointer<double>(cMem), cuda::getPointer<double>(aMem), cuda::getAlphaValue<double>(alpha),
							cuda::getBetaValue<double>(beta), dimensions, operation, cuda::getContext(context).getWorkspace().data<double>());
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

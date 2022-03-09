/*
 * reduction_utils.cuh
 *
 *  Created on: Mar 9, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef REDUCTION_UTILS_CUH_
#define REDUCTION_UTILS_CUH_

#include <backend_descriptors.hpp>

#include "numbers/numbers.cuh"

template<typename T>
struct limits
{
	__device__ T max() const
	{
		return scalar_zero<T>();
	}
};
template<>
struct limits<avocado::backend::float16>
{
	__device__ float max() const
	{
		return 65504;
	}
};
template<>
struct limits<avocado::backend::bfloat16>
{
	__device__ float max() const
	{
		return 3.4028e+38f;
	}
};
template<>
struct limits<float>
{
	__device__ float max() const
	{
		return 3.40282346e+38f;
	}
};
template<>
struct limits<double>
{
	__device__ double max() const
	{
		return 1.7976931348623157e+308;
	}
};

template<typename T>
class ReduceAdd
{
	numbers::Number<T> acc;
public:
	__device__ ReduceAdd() = default;
	__device__ ReduceAdd(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::zero<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc += x;
	}
	__device__ void combine_partial(const ReduceAdd &other)
	{
		acc += other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_add(acc);
	}
	__device__ ReduceAdd& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceMul
{
	numbers::Number<T> acc;
public:
	__device__ ReduceMul() = default;
	__device__ ReduceMul(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::one<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc *= x;
	}
	__device__ void combine_partial(const ReduceMul &other)
	{
		acc *= other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_mul(acc);
	}
	__device__ ReduceMul& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceMin
{
	numbers::Number<T> acc;
public:
	__device__ ReduceMin() = default;
	__device__ ReduceMin(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::Number<T>(limits<T>().max());
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		this->acc = numbers::min(this->acc, x);
	}
	__device__ void combine_partial(const ReduceMin &other)
	{
		this->acc = numbers::min(this->acc, other.acc);
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_min(acc);
	}
	__device__ ReduceMin& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceMax
{
	numbers::Number<T> acc;
public:
	__device__ ReduceMax() = default;
	__device__ ReduceMax(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::Number<T>(-limits<T>().max());
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		this->acc = numbers::max(this->acc, x);
	}
	__device__ void combine_partial(const ReduceMax &other)
	{
		this->acc = numbers::max(this->acc, other.acc);
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_max(acc);
	}
	__device__ ReduceMax& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceAMax
{
	numbers::Number<T> acc;
public:
	__device__ ReduceAMax() = default;
	__device__ ReduceAMax(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::zero<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc = numbers::max(acc, numbers::abs(x));
	}
	__device__ void combine_partial(const ReduceAMax &other)
	{
		acc = numbers::max(acc, other.acc);
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_max(acc);
	}
	__device__ ReduceAMax& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceNorm1
{
	numbers::Number<T> acc;
public:
	__device__ ReduceNorm1() = default;
	__device__ ReduceNorm1(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::zero<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc += numbers::abs(x);
	}
	__device__ void combine_partial(const ReduceNorm1 &other)
	{
		acc += other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_add(acc);
	}
	__device__ ReduceNorm1& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceNorm2
{
	numbers::Number<T> acc;
public:
	__device__ ReduceNorm2() = default;
	__device__ ReduceNorm2(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::zero<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc += numbers::square(x);
	}
	__device__ void combine_partial(const ReduceNorm2 &other)
	{
		acc += other.acc;
	}
	__device__ void final_action()
	{
		acc = numbers::sqrt(acc);
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_add(acc);
	}
	__device__ ReduceNorm2& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceMulNoZeros
{
	numbers::Number<T> acc;
public:
	__device__ ReduceMulNoZeros() = default;
	__device__ ReduceMulNoZeros(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::one<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		if (x != numbers::zero<T>())
			acc *= x;
	}
	__device__ void combine_partial(const ReduceMulNoZeros &other)
	{
		acc *= other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_mul(acc);
	}
	__device__ ReduceMulNoZeros& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceLogicalOR
{
	numbers::Number<T> acc;
public:
	__device__ ReduceLogicalOR() = default;
	__device__ ReduceLogicalOR(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::zero<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc |= x;
	}
	__device__ void combine_partial(const ReduceLogicalOR &other)
	{
		acc |= other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_or(acc);
	}
	__device__ ReduceLogicalOR& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};
template<typename T>
class ReduceLogicalAND
{
	numbers::Number<T> acc;
public:
	__device__ ReduceLogicalAND() = default;
	__device__ ReduceLogicalAND(numbers::Number<T> x) :
			acc(x)
	{
	}
	__device__ void init()
	{
		acc = numbers::one<T>();
	}
	__device__ void accumulate(numbers::Number<T> x)
	{
		acc &= x;
	}
	__device__ void combine_partial(const ReduceLogicalAND &other)
	{
		acc &= other.acc;
	}
	__device__ void final_action()
	{
	}
	__device__ void horizontal_reduction()
	{
		acc = horizontal_and(acc);
	}
	__device__ ReduceLogicalAND& operator=(numbers::Number<T> value)
	{
		acc = value;
		return *this;
	}
	__device__ operator numbers::Number<T>() const
	{
		return acc;
	}
};

template<typename T>
__device__ __host__ bool is_power_of_2(T x)
{
	return x > scalar_zero<T>() && !(x & (x - scalar_one<T>()));
}
template<typename T>
__device__ __host__ T round_to_power_of_2(T x)
{
	T result = 1;
	while (result <= x)
		result *= 2;
	return result / 2;
}

/*
 * Linear reduction (into a single element)
 */
template<class Accumulator>
__device__ void block_linear_reducion(Accumulator *ptr)
{
	assert(is_power_of_2(blockDim.x));
	for (uint32_t i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (threadIdx.x < i)
			ptr[threadIdx.x].combine_partial(ptr[threadIdx.x + i]);
		__syncthreads();
	}
}
template<class Accumulator, typename T>
__global__ void kernel_linear_reduction_step1(T *dst, const T* src, uint32_t elements)
{
	assert(blockDim.x == 1024);
	__shared__ Accumulator storage[1024];

	Accumulator acc;
	acc.init();
	for (uint32_t i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		acc.accumulate(numbers::Number<T>(src + i, elements - i));
	storage[threadIdx.x] = acc;

	__syncthreads();
	block_linear_reducion(storage);
	if (threadIdx.x == 0)
	{
		numbers::Number<T> tmp = (numbers::Number<T>) storage[0];
		tmp.store(dst + numbers::length<T>() * blockIdx.x);
	}
}
template<class Accumulator, typename T, typename U = T>
__global__ void kernel_linear_reduction_step2(T *dst, const T* src, U alpha, U beta)
{
	assert(blockDim.x <= 1024);
	__shared__ Accumulator storage[1024];
	for (int i = threadIdx.x; i < 1024; i += blockDim.x)
		storage[i].init();
	storage[threadIdx.x] = numbers::Number<T>(src + numbers::length<T>() * threadIdx.x);
	__syncthreads();
	block_linear_reducion(storage);

	if (threadIdx.x == 0)
	{
		storage[0].horizontal_reduction();
		storage[0].final_action();
		numbers::Number<T> tmp = numbers::Number<T>(alpha) * (numbers::Number<T>) storage[0];
		if (beta != scalar_zero<U>())
			tmp += numbers::Number<T>(beta) * numbers::Number<T>(dst, 1);
		tmp.store(dst, 1);
	}
}

template<class Accumulator, typename T, typename U = T>
avocado::backend::avStatus_t launch_linear_reduction(cudaStream_t stream, T* output, const T *input, const U alpha, const U beta, uint32_t elements,
		avocado::backend::cuda::MemoryDescriptor &workspace)
{
	assert(output != nullptr);
	assert(input != nullptr);
	dim3 blockDim(1024);
	const int partial_results = round_to_power_of_2(gridSize<1024>(elements, blockDim.x));
	if (workspace.size() < sizeof(T) * partial_results)
		return avocado::backend::AVOCADO_STATUS_INTERNAL_ERROR;

	kernel_linear_reduction_step1<Accumulator, T> <<<partial_results, blockDim, 0, stream>>>(workspace.data<T>(), input, elements);
	kernel_linear_reduction_step2<Accumulator, T, U> <<<1, partial_results, 0, stream>>>(output, workspace.data<T>(), alpha, beta);
	return checkForErrors();
}

/*
 * Broadcasted reduction (into 1D array)
 */
template<class Accumulator>
__device__ void block_broadcasted_reduction(Accumulator *ptr) noexcept
{
	assert(blockDim.x == 32);
	assert(blockDim.y == 32);
	for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (threadIdx.y < i)
			ptr[threadIdx.y * 32 + threadIdx.x].combine_partial(ptr[(i + threadIdx.y) * 32 + threadIdx.x]);
		__syncthreads();
	}
}
template<class Accumulator, typename T>
__global__ void kernel_broadcasted_reduction_step1(T *dst, const T* src, uint32_t first_dim, uint32_t last_dim)
{
	assert(blockDim.x * blockDim.y == 1024);
	__shared__ Accumulator storage[32 * 32];
	for (uint32_t j = numbers::length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += numbers::length<T>() * blockDim.x * gridDim.x)
	{
		uint32_t idx = j + numbers::length<T>() * threadIdx.x;

		Accumulator acc;
		acc.init();
		if (idx < last_dim)
		{
			for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				acc.accumulate(numbers::Number<T>(src + i * last_dim + idx, last_dim - idx));
		}
		storage[threadIdx.y * 32 + threadIdx.x] = acc;

		__syncthreads();
		block_broadcasted_reduction(storage);
		if (threadIdx.y == 0 and idx < last_dim)
		{
			numbers::Number<T> tmp = (numbers::Number<T>) (storage[0 * 32 + threadIdx.x]);
			tmp.store(dst + blockIdx.y * last_dim + idx, last_dim - idx);
		}
	}
}
template<class Accumulator, typename T, typename U = T>
__global__ void kernel_broadcasted_reduction_step2(T *dst, const T* src, U alpha, U beta, uint32_t first_dim, uint32_t last_dim)
{
	assert(blockDim.x * blockDim.y == 1024);
	__shared__ Accumulator storage[32 * 32];
	for (uint32_t j = numbers::length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += numbers::length<T>() * blockDim.x * gridDim.x)
	{
		uint32_t idx = j + numbers::length<T>() * threadIdx.x;

		Accumulator acc;
		acc.init();
		if (idx < last_dim)
		{
			for (uint32_t i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				numbers::Number<T> tmp = numbers::Number<T>(src + i * last_dim + idx, last_dim - idx);
				acc.combine_partial(tmp);
			}
		}
		storage[threadIdx.y * 32 + threadIdx.x] = acc;

		__syncthreads();
		block_broadcasted_reduction(storage);
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

template<class Accumulator, typename T, typename U = T>
avocado::backend::avStatus_t launch_broadcasted_reduction(cudaStream_t stream, T* output, const T *input, const U alpha, const U beta, uint32_t firstDim,
		uint32_t lastDim, avocado::backend::cuda::MemoryDescriptor &workspace)
{
	assert(output != nullptr);
	assert(input != nullptr);
	if (lastDim > 1024)
		return avocado::backend::AVOCADO_STATUS_BAD_PARAM;

	dim3 blockDim(32, 32);

	int grid_dim_x = gridSize<32>(firstDim, blockDim.x);
	int grid_dim_y = gridSize<128>(lastDim, blockDim.y);
	if (workspace.size() < sizeof(T) * blockDim.x * grid_dim_x * grid_dim_y)
		return avocado::backend::AVOCADO_STATUS_INTERNAL_ERROR;

	dim3 gridDim1(grid_dim_x, grid_dim_y);
	kernel_broadcasted_reduction_step1<Accumulator, T> <<<gridDim1, blockDim, 0, stream>>>(workspace.data<T>(), input, firstDim, lastDim);

	dim3 gridDim2(grid_dim_x, 1);
	kernel_broadcasted_reduction_step2<Accumulator, T, U> <<<gridDim2, blockDim, 0, stream>>>(output, workspace.data<T>(), alpha, beta, grid_dim_y, lastDim);
	return checkForErrors();
}

#endif /* REDUCTION_UTILS_CUH_ */

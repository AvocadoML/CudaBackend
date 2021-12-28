//============================================================================
// Name        : CudaBackend.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cuda_runtime_api.h>
#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include <limits>

#include "../src/utilities.hpp"
#include "../src/activations.cuh"
using namespace avocado::backend;

class TensorWrapper
{
private:
	avTensorDescriptor_t desc;
	avMemoryDescriptor_t mem;
public:
	TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype, int device)
	{
		cudaCreateTensorDescriptor(&desc);
		cudaSetTensorDescriptor(desc, dtype, dimensions.size(), dimensions.begin());

		avSize_t size_in_bytes = getTensor(desc).sizeInBytes();
		cudaCreateMemoryDescriptor(&mem, device, size_in_bytes);
		cudaSetMemory(device, mem, size_in_bytes, nullptr, 0);
	}
	~TensorWrapper()
	{
		cudaDestroyTensorDescriptor(desc);
		cudaDestroyMemoryDescriptor(mem);
	}
	template<typename T>
	void fill(T value)
	{
		assert(typeOf<T>() == getTensor(desc).dtype());
		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(getTensor(desc).volume());
		for (avSize_t i = 0; i < getTensor(desc).volume(); i++)
			tmp[i] = value;
		cudaMemcpy(getPointer(mem), tmp.get(), getTensor(desc).sizeInBytes(), cudaMemcpyHostToDevice);
	}
	template<typename T>
	void set(T value, std::initializer_list<int> idx)
	{
		cudaMemcpy(getPointer<T>(mem) + getTensor(desc).getIndex(idx), &value, sizeof(T), cudaMemcpyHostToDevice);
	}
	template<typename T>
	T get(std::initializer_list<int> idx) const
	{
		T result;
		cudaMemcpy(&result, getPointer<T>(mem) + getTensor(desc).getIndex(idx), sizeof(T), cudaMemcpyDeviceToHost);
		return result;
	}
	operator avTensorDescriptor_t() noexcept
	{
		return desc;
	}
	template<typename T>
	T* data() noexcept
	{
		return getPointer<T>(mem);
	}
	template<typename T>
	const T* data() const noexcept
	{
		return getPointer<T>(mem);
	}
};

template<typename T>
struct limits
{
	__device__ T max() const noexcept
	{
	}
	__device__ T lowest() const noexcept
	{
	}
};
template<>
struct limits<half>
{
	__device__ float max() const noexcept
	{
		return 65504;
	}
	__device__ float lowest() const noexcept
	{
		return -max();
	}
};
template<>
struct limits<float>
{
	__device__ float max() const noexcept
	{
		return 3.40282346638528859811704183484516925e+38f;
	}
	__device__ float lowest() const noexcept
	{
		return -max();
	}
};
template<>
struct limits<double>
{
	__device__ double max() const noexcept
	{
		return 1.79769313486231570814527423731704357e+308;
	}
	__device__ double lowest() const noexcept
	{
		return -max();
	}
};

template<typename T>
class ReduceAdd
{
	T acc = zero<T>();
public:
	__device__ ReduceAdd() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc += x;
	}
	__device__ void combine_partial(ReduceAdd other) noexcept
	{
		acc += other.acc;
	}
	__device__ ReduceAdd& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceMul
{
	T acc = one<T>();
public:
	__device__ ReduceMul() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc *= x;
	}
	__device__ void combine_partial(ReduceMul other) noexcept
	{
		acc *= other.acc;
	}
	__device__ ReduceMul& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceMin
{
	T acc = limits<T>().max();
public:
	__device__ ReduceMin() = default;
	__device__ void accumulate(T x) noexcept
	{
		this->acc = min(this->acc, x);
	}
	__device__ void combine_partial(ReduceMin other) noexcept
	{
		this->acc = min(this->acc, other.acc);
	}
	__device__ ReduceMin& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceMax
{
	T acc = limits<T>().lowest();
public:
	__device__ ReduceMax() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc = max(acc, x);
	}
	__device__ void combine_partial(ReduceMax other) noexcept
	{
		acc = max(acc, other.acc);
	}
	__device__ ReduceMax& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceAMax
{
	T acc = zero<T>();
public:
	__device__ ReduceAMax() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc = max(acc, abs(x));
	}
	__device__ void combine_partial(ReduceAMax other) noexcept
	{
		acc = max(acc, other.acc);
	}
	__device__ ReduceAMax& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceNorm1
{
	T acc = zero<T>();
public:
	__device__ ReduceNorm1() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc += abs(x);
	}
	__device__ void combine_partial(ReduceNorm1 other) noexcept
	{
		acc += other.acc;
	}
	__device__ ReduceNorm1& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceNorm2
{
	T acc = zero<T>();
public:
	__device__ ReduceNorm2() = default;
	__device__ void accumulate(T x) noexcept
	{
		acc += square(x);
	}
	__device__ void combine_partial(ReduceNorm2 other) noexcept
	{
		acc += other.acc;
	}
	__device__ ReduceNorm2& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};
template<typename T>
class ReduceMulNoZeros
{
	T acc = one<T>();
public:
	__device__ ReduceMulNoZeros() = default;
	__device__ void accumulate(T x) noexcept
	{
		if (x != zero<T>())
			acc *= x;
	}
	__device__ void combine_partial(ReduceMulNoZeros other) noexcept
	{
		acc *= other.acc;
	}
	__device__ ReduceMulNoZeros& operator=(T value) noexcept
	{
		acc = value;
		return *this;
	}
	__device__ operator T() const noexcept
	{
		return acc;
	}
};

template<class Acc>
__device__ void reduce_linear_within_block(Acc *ptr) noexcept
{
	assert(ispow2(blockDim.x));
	for (unsigned int i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (threadIdx.x < i)
			ptr[threadIdx.x].combine_partial(ptr[threadIdx.x + i]);
		__syncthreads();
	}
}
template<class Acc, typename T>
__global__ void kernel_reduce_linear_1(T *dst, const T* src, unsigned int elements)
{
	__shared__ Acc storage[1024];

	Acc acc;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
		acc.accumulate(src[i]);

	storage[threadIdx.x] = acc;

	__syncthreads();
	reduce_linear_within_block(storage);
	if (threadIdx.x == 0)
		dst[blockIdx.x] = storage[0];
}
template<class Acc, typename T>
__global__ void kernel_reduce_linear_2(T *dst, const T* src, T alpha, T beta)
{
	__shared__ Acc storage[1024];
	storage[threadIdx.x] = src[threadIdx.x];
	__syncthreads();
	reduce_linear_within_block(storage);
	if (threadIdx.x == 0)
	{
		if (beta == zero<T>())
			dst[0] = alpha * storage[0];
		else
			dst[0] = alpha * storage[0] + beta * dst[0];
	}
}

template<class Acc>
__device__ void reduce_broadcasted_within_block(Acc *ptr) noexcept
{
	for (int i = 16; i >= 1; i /= 2) // sum results stored in temporary array
	{
		if (threadIdx.y < i)
			ptr[threadIdx.y * 32 + threadIdx.x].combine_partial(ptr[(i + threadIdx.y) * 32 + threadIdx.x]);
		__syncthreads();
	}
}
template<class Acc, typename T>
__global__ void kernel_reduce_broadcasted_1(T *dst, const T* src, unsigned int first_dim, unsigned int last_dim)
{
	__shared__ Acc storage[32][32];
	for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
	{
		unsigned int idx = j + threadIdx.x;

		Acc acc;
		if (idx < last_dim)
		{
			for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				acc.accumulate(src[i * last_dim + idx]);
		}
		storage[threadIdx.y][threadIdx.x] = acc;

		__syncthreads();
		reduce_broadcasted_within_block(reinterpret_cast<Acc*>(storage));
		if (threadIdx.y == 0 and idx < last_dim)
			dst[blockIdx.y * last_dim + idx] = storage[0][threadIdx.x];
	}
}
template<class Acc, typename T>
__global__ void kernel_reduce_broadcasted_2(T *dst, const T* src, T alpha, T beta, unsigned int first_dim, unsigned int last_dim)
{
	__shared__ Acc storage[32][32];
	for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
	{
		unsigned int idx = j + threadIdx.x;

		Acc acc;
		if (idx < last_dim)
		{
			for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				acc.combine_partial(reinterpret_cast<const Acc*>(src)[i * last_dim + idx]);
		}
		storage[threadIdx.y][threadIdx.x] = acc;

		__syncthreads();
		reduce_broadcasted_within_block(reinterpret_cast<Acc*>(storage));
		if (threadIdx.y == 0 and idx < last_dim)
		{
			if (beta == zero<T>())
				dst[blockIdx.y * last_dim + idx] = alpha * storage[0][threadIdx.x];
			else
				dst[blockIdx.y * last_dim + idx] = alpha * storage[0][threadIdx.x] + beta * dst[blockIdx.y * last_dim + idx];
		}
	}
}

int main()
{
	std::cout << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << '\n';
//	avContext_t context;
//	cudaCreateContext(&context, 0);

//	TensorWrapper dst( { 128 }, typeOf<float>(), 0);
//	TensorWrapper input( { 128, 20, 20, 120 }, typeOf<float>(), 0);
//	TensorWrapper workspace( { 128, 120 }, typeOf<float>(), 0);

//	input.fill(1.0f);
//	input.set(-10.0f, { 5, 0, 0, 4 });

//	dim3 blockDim(32, 32);
//	dim3 gridDim1(4, 128);
//	dim3 gridDim2(4, 1);

//	int partial_results = 64; // must be power of 2
//	kernel_reduce_linear_1<ReduceMin<float>, float> <<<partial_results, 1024, 0, get_stream(context)>>>(workspace.data<float>(), input.data<float>(),
//			volume(input));
//	kernel_reduce_linear_2<ReduceMin<float>, float> <<<1, partial_results, 0, get_stream(context)>>>(dst.data<float>(), workspace.data<float>(), 1.0f, 0.0f);

//	kernel_reduce_broadcasted_1<ReduceMin<float>, float> <<<gridDim1, blockDim, 0, get_stream(context)>>>(workspace.data<float>(), input.data<float>(),
//			volumeWithoutLastDim(input), lastDim(input));
//	kernel_reduce_broadcasted_2<ReduceMin<float>, float> <<<gridDim2, blockDim, 0, get_stream(context)>>>(dst.data<float>(), workspace.data<float>(), 1.0f,
//			0.0f, firstDim(workspace), lastDim(workspace));
//	cudaStreamSynchronize(get_stream(context));

//	for (int i = 0; i < 1; i++)
//	{
//		for (int j = 0; j < 128; j++)
//			std::cout << dst.get<float>( { j }) << ' ';
//		std::cout << '\n';
//	}

//	cudaDestroyContext(context);

	std::cout << "END" << std::endl;
	return 0;
}

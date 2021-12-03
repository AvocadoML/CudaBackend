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
#include <avocado/backend/tensor_helpers.hpp>
#include "../src/context.hpp"
#include "../src/utilities.hpp"
#include "../src/activations.cuh"
using namespace avocado::backend;

class TensorWrapper
{
private:
	TensorDescriptor desc;
public:
	TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype, int device)
	{
		desc.shape = createShapeDescriptor(dimensions);
		avSize_t size_in_bytes = volume(desc.shape) * dataTypeSize(dtype);
		cudaSetDevice(device);
		cudaMalloc(&(desc.data), size_in_bytes);
		cudaMemset(desc.data, 0, size_in_bytes);
		desc.dtype = dtype;
	}
	~TensorWrapper()
	{
		cudaFree(desc.data);
	}
	template<typename T>
	void fill(T value)
	{
		assert(typeOf<T>() == desc.dtype);
		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(volume(desc.shape));
		for (avSize_t i = 0; i < volume(desc.shape); i++)
			tmp[i] = value;
		cudaMemcpy(desc.data, tmp.get(), volume(desc.shape) * dataTypeSize(desc.dtype), cudaMemcpyHostToDevice);
	}
	template<typename T>
	T& at(std::initializer_list<int> idx)
	{
		return *(reinterpret_cast<T*>(desc.data) + get_index(idx));
	}
	template<typename T>
	T get(std::initializer_list<int> idx) const
	{
		T result;
		cudaMemcpy(&result, reinterpret_cast<const T*>(desc.data) + get_index(idx), sizeof(T), cudaMemcpyDeviceToHost);
		return result;
	}
	operator avTensor_t() noexcept
	{
		return &desc;
	}
	template<typename T>
	T* data() noexcept
	{
		return reinterpret_cast<T*>(desc.data);
	}
	template<typename T>
	const T* data() const noexcept
	{
		return reinterpret_cast<const T*>(desc.data);
	}
private:
	avSize_t get_index(std::initializer_list<int> index) const
	{
		assert(desc.shape.length == static_cast<int>(index.size()));
		avSize_t result = 0;
		avSize_t tmp = 1;
		for (int i = index.size() - 1; i >= 0; i--)
		{
			int idx = index.begin()[i];
			assert(idx >= 0 && idx < desc.shape.dim[i]);
			result += idx * tmp;
			tmp *= desc.shape.dim[i];
		}
		return result;
	}
};

template<typename T>
class OpAdd
{
public:
	__device__ T operator()(T lhs, T rhs) const noexcept
	{
		return lhs + rhs;
	}
};
template<typename T>
class OpSub
{
public:
	__device__ T operator()(T lhs, T rhs) const noexcept
	{
		return lhs - rhs;
	}
};
template<typename T>
class OpMul
{
public:
	__device__ T operator()(T lhs, T rhs) const noexcept
	{
		return lhs * rhs;
	}
};
template<typename T>
class OpMin
{
public:
	__device__ T operator()(T lhs, T rhs) const noexcept
	{
		return min(lhs, rhs);
	}
};
template<typename T>
class OpMax
{
public:
	__device__ T operator()(T lhs, T rhs) const noexcept
	{
		return max(lhs, rhs);
	}
};

template<typename T>
class OpAbs
{
public:
	__device__ T operator()(T x) const noexcept
	{
		return abs(x);
	}
};
template<typename T>
class OpSquare
{
public:
	__device__ T operator()(T x) const noexcept
	{
		return x * x;
	}
};
template<typename T>
class OpSqrt
{
public:
	__device__ T operator()(T x) const noexcept
	{
		return sqrt(x);
	}
};
template<typename T>
class OpNot
{
public:
	__device__ T operator()(T x) const noexcept
	{
		return -x;
	}
};

template<class Op, typename T>
__global__ void kernel_op_single_tensor(T* dst, const T *input, const T alpha, const unsigned int elements)
{
	Op operation;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
		dst[i] = operation(input[i] * alpha);
}
template<class Op, typename T>
__global__ void kernel_op_single_tensor(T* dst, const T *input, const T alpha, const T beta, const unsigned int elements)
{
	Op operation;
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
		dst[i] = operation(input[i] * alpha) + beta * dst[i];
}

template<typename T>
__device__ T perform_op(avOpTensorOp_t operation, T lhs, T rhs) noexcept
{
	switch (operation)
	{
		case AVOCADO_OP_TENSOR_ADD:
			return lhs + rhs;
		case AVOCADO_OP_TENSOR_SUB:
			return lhs - rhs;
		case AVOCADO_OP_TENSOR_MUL:
			return lhs * rhs;
		case AVOCADO_OP_TENSOR_MIN:
			return min(lhs, rhs);
		case AVOCADO_OP_TENSOR_MAX:
			return max(lhs, rhs);
		default:
			return zero<T>();
	}
}

__device__ constexpr bool uses_single_operand(avOpTensorOp_t operation) noexcept
{
//	return operation == AVOCADO_OP_TENSOR_SQRT or operation == AVOCADO_OP_TENSOR_NOT;
}

template<class Op, typename T>
__global__ void template_kernel(T* dst, const T *input1, T alpha1, const T *input2, T alpha2, T beta, unsigned int first_dim, unsigned int last_dim)
{
	Op operation;
	extern __shared__ float stored_input2[];

	for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
	{
		int tmp_idx = j + threadIdx.x;
		if (threadIdx.y == 0 and tmp_idx < last_dim)
			stored_input2[threadIdx.x] = input2[tmp_idx] * alpha2;
		__syncthreads();
		if (tmp_idx < last_dim)
			for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				T lhs = input1[i * last_dim + tmp_idx] * alpha1;
				T rhs = stored_input2[threadIdx.x];
				T tmp = operation(lhs, rhs);
				if (beta != zero<T>())
					tmp += beta * dst[i * last_dim + tmp_idx];
				dst[i * last_dim + tmp_idx] = tmp;
			}
		__syncthreads();
	}
}
template<class Op, typename T>
__global__ void template_kernel_single(T* dst, const T *input1, T alpha1, const T *input2, T alpha2, T beta, unsigned int elements)
{
	Op operation;
	float stored_input2 = input2[0] * alpha2;

	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
	{
		T lhs = input1[i] * alpha1;
		T tmp = operation(lhs, stored_input2);
		if (beta != zero<T>())
			tmp += beta * dst[i];
		dst[i] = tmp;
	}
}

template<typename T>
__global__ void template_kernel(T* dst, const T *input1, T alpha1, const T *input2, T alpha2, T beta, unsigned int first_dim, unsigned int last_dim,
		avOpTensorOp_t operation)
{
	extern __shared__ float stored_input2[];

	for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
	{
		int tmp_idx = j + threadIdx.x;
//		if (not uses_single_operand(operation))
//		{
		if (threadIdx.y == 0 and tmp_idx < last_dim)
			stored_input2[threadIdx.x] = input2[tmp_idx] * alpha2;
		__syncthreads();
//		}
		if (tmp_idx < last_dim)
			for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
			{
				T lhs = input1[i * last_dim + tmp_idx] * alpha1;
				T rhs = uses_single_operand(operation) ? zero<T>() : stored_input2[threadIdx.x];
//				T rhs = stored_input2[threadIdx.x];
//				T tmp = perform_op(operation, lhs, rhs);
				T tmp = max(lhs, rhs);
				if (beta != zero<T>())
					tmp += beta * dst[i * last_dim + tmp_idx];
				dst[i * last_dim + tmp_idx] = tmp;
			}
//		if (not uses_single_operand(operation))
		__syncthreads();
	}
}
template<typename T>
__global__ void template_kernel(T* dst, const T *input1, T alpha1, T beta, unsigned int elements, avOpTensorOp_t operation)
{
	for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
	{
		T lhs = input1[i] * alpha1;
		T tmp = perform_op(operation, lhs, 0.0f);
		if (beta != zero<T>())
			tmp += beta * dst[i];
		dst[i] = tmp;
	}
}

int main()
{
	std::cout << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << '\n';
	avContext_t context;
	cudaCreateContext(&context, 0);

	TensorWrapper dst( { 128, 20, 20, 128 }, typeOf<float>(), 0);
	TensorWrapper input1( { 128, 20, 20, 128 }, typeOf<float>(), 0);
	TensorWrapper input2( { 128 }, typeOf<float>(), 0);

	input1.fill(2.0f);
	input2.fill(0.22f);

	dim3 blockDim(256);
	dim3 gridDim(28 * 80);
	kernel_op_single_tensor<OpAbs<float>, float> <<<gridDim, blockDim, 0, get_stream(context)>>>(dst.data<float>(), input1.data<float>(), 1.0f,
			volume(input1));
	cudaStreamSynchronize(get_stream(context));

//	dim3 blockDim(32, 32);
//	dim3 gridDim(4, 28 * 10);

//	template_kernel<OpAdd<float>, float> <<<gridDim, blockDim, sizeof(float) * blockDim.x, get_stream(context)>>>(dst.data<float>(), input1.data<float>(), 1.0f,
//			input2.data<float>(), 1.0f, 0.0f, volumeWithoutLastDim(input1), lastDim(input1));
//	cudaStreamSynchronize(get_stream(context));

//	dim3 blockDim(1024);
//	dim3 gridDim(28 * 10);
//	template_kernel_single<OpAdd<float>, float> <<<gridDim, blockDim, sizeof(float) * blockDim.x, get_stream(context)>>>(dst.data<float>(),
//			input1.data<float>(), 1.0f, input2.data<float>(), 1.0f, 0.0f, volume(input1));
//	cudaStreamSynchronize(get_stream(context));

//	template_kernel<<<gridDim, blockDim, sizeof(float) * blockDim.x, get_stream(context)>>>(input1.data<float>(), input1.data<float>(), 1.0f,
//			input2.data<float>(), 1.0f, 0.0f, volumeWithoutLastDim(input1), lastDim(input1), AVOCADO_OP_TENSOR_MAX);
//	cudaStreamSynchronize(get_stream(context));

//	for (int i = 0; i < 2; i++)
//	{
//		for (int j = 0; j < 126; j++)
//			std::cout << dst.get<float>( { i, j }) << ' ';
//		std::cout << '\n';
//	}

	cudaDestroyContext(context);

	std::cout << "END" << std::endl;
	return 0;
}

/*
 * tensor_op.cu
 *
 *  Created on: Dec 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "activations.cuh"
#include "logical_ops.cuh"
#include "utilities.hpp"
#include "numbers/numbers.cuh"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::BACKEND_NAMESPACE;

	template<typename T>
	class BinaryOpAdd
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return lhs + rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs + rhs;
//		}
	};
	template<typename T>
	class BinaryOpAddSquare
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return lhs + numbers::square(rhs);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs + square(rhs);
//		}
	};
	template<typename T>
	class BinaryOpSub
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return lhs - rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs - rhs;
//		}
	};
	template<typename T>
	class BinaryOpMul
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return lhs * rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs * rhs;
//		}
	};
	template<typename T>
	class BinaryOpDiv
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return lhs / rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs / rhs;
//		}
	};
	template<typename T>
	class BinaryOpMod
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::mod(lhs, rhs);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return fmodf(lhs, rhs);
//		}
	};
	template<typename T>
	class BinaryOpPow
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::pow(lhs, rhs);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return powf(lhs, rhs);
//		}
	};
	template<typename T>
	class BinaryOpMin
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::min(lhs, rhs);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return min(lhs, rhs);
//		}
	};
	template<typename T>
	class BinaryOpMax
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::max(lhs, rhs);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return max(lhs, rhs);
//		}
	};
	template<typename T>
	class BinaryOpCompareEq
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs == rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs == rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpCompareNeq
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs != rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs != rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpCompareGt
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs > rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs > rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpCompareGe
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs >= rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs >= rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpCompareLt
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs < rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs < rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpCompareLe
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return (lhs <= rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return (lhs <= rhs) ? bit_cast<T>(-1) : bit_cast<T>(0);
//		}
	};
	template<typename T>
	class BinaryOpLogicalAnd
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return lhs & rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs & rhs;
//		}
	};
	template<typename T>
	class BinaryOpLogicalOr
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return lhs | rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs | rhs;
//		}
	};
	template<typename T>
	class BinaryOpLogicalXor
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> lhs, numbers::Number<T> rhs) const noexcept
		{
			return numbers::Number<T>();
//			return lhs ^ rhs;
		}
//		__device__ T operator()(T lhs, T rhs) const noexcept
//		{
//			return lhs ^ rhs;
//		}
	};

	/*
	 *
	 * Logical operations
	 *
	 */

	/**
	 * \brief Kernel when input1 and input2 have the same shape.
	 */
	template<class Op, typename T>
	__global__ void kernel_binary_logical_op_same_shape(T* dst, const T *input1, const T *input2, unsigned int elements)
	{
//		Op operation;
//		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
//			dst[i] = operation(input1[i], input2[i]);
	}
	/**
	 * \brief Kernel when input2 is broadcasted into input1 but is not a single element tensor.
	 */
	template<class Op, typename T>
	__global__ void kernel_binary_logical_op_broadcasted(T* dst, const T *input1, const T *input2, unsigned int first_dim, unsigned int last_dim)
	{
//		__shared__ T stored_input2[128];
//
//		Op operation;
//		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
//		{
//			int tmp_idx = j + threadIdx.x;
//			if (threadIdx.y == 0 and tmp_idx < last_dim)
//				stored_input2[threadIdx.x] = input2[tmp_idx];
//			__syncthreads();
//			if (tmp_idx < last_dim)
//				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//					dst[i * last_dim + tmp_idx] = operation(input1[i * last_dim + tmp_idx], stored_input2[threadIdx.x]);
//			__syncthreads();
//		}
	}
	/**
	 * \brief Kernel when input2 has only one element that is broadcasted into input1.
	 */
	template<class Op, typename T>
	__global__ void kernel_binary_logical_op_single_element(T* dst, const T *input1, const T *input2, unsigned int elements)
	{
//		Op operation;
//		T rhs = input2[0];
//		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
//			dst[i] = operation(input1[i], rhs);
	}
	template<class Op, typename T>
	void helper_binary_logical_op(cudaStream_t stream, T* dst, const T *input1, const T* input2, BroadcastedDimensions dimensions)
	{
		if (dimensions.first == 1) // both input1 and input2 have the same shape
		{
			dim3 blockDim(256);
			dim3 gridDim = gridSize<1024>(dimensions.last, blockDim.x);
			kernel_binary_logical_op_same_shape<Op, T> <<<gridDim, blockDim, 0, stream>>>(dst, input1, input2, dimensions.last);
		}
		else
		{
			if (dimensions.last == 1) // input2 is a single element
			{
				dim3 blockDim(256);
				dim3 gridDim = gridSize<1024>(dimensions.first, blockDim.x);
				kernel_binary_logical_op_single_element<Op, T> <<<gridDim, blockDim, 0, stream>>>(dst, input1, input2, dimensions.first);
			}
			else
			{
				dim3 blockDim(32, 32);
				dim3 gridDim(8, 128);
				kernel_binary_logical_op_broadcasted<Op, T> <<<gridDim, blockDim, 0, stream>>>(dst, input1, input2, dimensions.first, dimensions.last);
			}
		}
	}
	template<typename T>
	avStatus_t launcher_binary_logical_op(cudaStream_t stream, T* dst, const T *input1, const T* input2, BroadcastedDimensions dimensions,
			avBinaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_BINARY_OP_LOGICAL_AND:
				helper_binary_logical_op<BinaryOpLogicalAnd<T>, T>(stream, dst, input1, input2, dimensions);
				break;
			case AVOCADO_BINARY_OP_LOGICAL_OR:
				helper_binary_logical_op<BinaryOpLogicalOr<T>, T>(stream, dst, input1, input2, dimensions);
				break;
			case AVOCADO_BINARY_OP_LOGICAL_XOR:
				helper_binary_logical_op<BinaryOpLogicalXor<T>, T>(stream, dst, input1, input2, dimensions);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return checkForErrors();
	}

	/*
	 *
	 *  Arithmetic operations
	 *
	 */

	/**
	 * \brief Kernel when input1 and input2 have the same shape.
	 */
	template<class Op, typename T, typename U = T>
	__global__ void kernel_binary_op_same_shape(T* dst, const T *input1, U alpha1, const T *input2, U alpha2, U beta, unsigned int elements)
	{
		numbers::Number<T> _alpha1(alpha1);
		numbers::Number<T> _alpha2(alpha2);
		numbers::Number<T> _beta(beta);
		Op operation;
//		Store<T, U> store;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> lhs(input1 + i, elements - i);
			numbers::Number<T> rhs(input2 + i, elements - i);
			numbers::Number<T> tmp = operation(_alpha1 * lhs, _alpha2 * rhs);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(dst + i, elements - i);
			tmp.store(dst + i, elements - i);

//			U lhs = alpha1 * input1[i];
//			U rhs = alpha2 * input2[i];
//			U tmp = operation(lhs, rhs);
//			if (beta != zero<U>())
//				tmp += beta * dst[i];
//			dst[i] = store(tmp);
		}
	}
	/**
	 * \brief Kernel when input2 is broadcasted into input1 but is not a single element tensor.
	 */
	template<class Op, typename T, typename U = T>
	__global__ void kernel_binary_op_broadcasted(T* dst, const T *input1, U alpha1, const T *input2, U alpha2, U beta, unsigned int first_dim,
			unsigned int last_dim)
	{
		__shared__ numbers::Number<T> stored_input2[128];

		numbers::Number<T> _alpha1(alpha1);
		numbers::Number<T> _alpha2(alpha2);
		numbers::Number<T> _beta(beta);
		Op operation;
		for (unsigned int j = numbers::length<T>() * blockIdx.x * blockDim.x; j < last_dim; j += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			int tmp_idx = j + numbers::length<T>() * threadIdx.x;
			if (threadIdx.y == 0 and tmp_idx < last_dim)
				stored_input2[threadIdx.x] = _alpha2 * numbers::Number<T>(input2 + tmp_idx, last_dim - tmp_idx);
			__syncthreads();
			if (tmp_idx < last_dim)
				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
				{
					numbers::Number<T> lhs = _alpha1 * numbers::Number<T>(input1 + i * last_dim + tmp_idx, last_dim - tmp_idx);
					numbers::Number<T> rhs = stored_input2[threadIdx.x];
					numbers::Number<T> tmp = operation(lhs, rhs);
					if (_beta != numbers::zero<T>())
						tmp += _beta * numbers::Number<T>(dst + i * last_dim + tmp_idx, last_dim - tmp_idx);
					tmp.store(dst + i * last_dim + tmp_idx, last_dim - tmp_idx);
				}
			__syncthreads();
		}

//		extern __shared__ ComputeType stored_input2[];
//		__shared__ U stored_input2[128];
//
//		numbers::Number<T> _alpha1(alpha1);
//		numbers::Number<T> _alpha2(alpha2);
//		numbers::Number<T> _beta(beta);
//		Op operation;
//		Store<T, U> store;
//		for (unsigned int j = blockIdx.x * blockDim.x; j < last_dim; j += blockDim.x * gridDim.x)
//		{
//			int tmp_idx = j + threadIdx.x;
//			if (threadIdx.y == 0 and tmp_idx < last_dim)
//				stored_input2[threadIdx.x] = alpha2 * input2[tmp_idx];
//			__syncthreads();
//			if (tmp_idx < last_dim)
//				for (unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; i < first_dim; i += blockDim.y * gridDim.y)
//				{
//					U lhs = alpha1 * input1[i * last_dim + tmp_idx];
//					U rhs = stored_input2[threadIdx.x];
//					U tmp = operation(lhs, rhs);
//					if (beta != zero<U>())
//						tmp += beta * dst[i * last_dim + tmp_idx];
//					dst[i * last_dim + tmp_idx] = store(tmp);
//				}
//			__syncthreads();
//		}
	}
	/**
	 * \brief Kernel when input2 has only one element that is broadcasted into input1.
	 */
	template<class Op, typename T, typename U = T>
	__global__ void kernel_binary_op_single_element(T* dst, const T *input1, U alpha1, const T *input2, U alpha2, U beta, unsigned int elements)
	{
		numbers::Number<T> _alpha1(alpha1);
		numbers::Number<T> _alpha2(alpha2);
		numbers::Number<T> _beta(beta);
		Op operation;
		numbers::Number<T> rhs = _alpha2 * numbers::Number<T>(input2[0]);
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> lhs = _alpha1 * numbers::Number<T>(input1 + i, elements - i);
			numbers::Number<T> tmp = operation(lhs, rhs);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(dst + i, elements - i);
			tmp.store(dst + i, elements - i);
		}

//		Op operation;
//		Store<T, U> store;
//		T rhs = alpha2 * input2[0];
//		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
//		{
//			U lhs = alpha1 * input1[i];
//			U tmp = operation(lhs, rhs);
//			if (beta != zero<U>())
//				tmp += beta * dst[i];
//			dst[i] = store(tmp);
//		}
	}

	template<class Op, typename T, typename U = T>
	void helper_binary_op(cudaStream_t stream, T* dst, const T *input1, const U alpha1, const T* input2, const U alpha2, const U beta,
			BroadcastedDimensions dimensions)
	{
		if (dimensions.first == 1) // both input1 and input2 have the same shape
		{
			dim3 blockDim(256);
			dim3 gridDim = gridSize<1024>(dimensions.last, blockDim.x);
			kernel_binary_op_same_shape<Op, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input1, alpha1, input2, alpha2, beta, dimensions.last);
		}
		else
		{
			if (dimensions.last == 1) // input2 is a single element
			{
				dim3 blockDim(256);
				dim3 gridDim = gridSize<1024>(dimensions.first, blockDim.x);
				kernel_binary_op_single_element<Op, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input1, alpha1, input2, alpha2, beta, dimensions.first);
			}
			else
			{
				dim3 blockDim(32, 32);
				dim3 gridDim(8, 128);
				kernel_binary_op_broadcasted<Op, T, U> <<<gridDim, blockDim, 0 /*sizeof(U) * blockDim.x*/, stream>>>(dst, input1, alpha1, input2, alpha2, beta,
						dimensions.first, dimensions.last);
			}
		}
	}
	template<typename T, typename U = T>
	avStatus_t launcher_binary_op(cudaStream_t stream, T* dst, const T *input1, const U alpha1, const T* input2, const U alpha2, const U beta,
			BroadcastedDimensions dimensions, avBinaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_BINARY_OP_ADD:
				helper_binary_op<BinaryOpAdd<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_ADD_SQUARE:
				helper_binary_op<BinaryOpAddSquare<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_SUB:
				helper_binary_op<BinaryOpSub<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MUL:
				helper_binary_op<BinaryOpMul<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_DIV:
				helper_binary_op<BinaryOpDiv<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MOD:
				helper_binary_op<BinaryOpMod<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_POW:
				helper_binary_op<BinaryOpPow<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MIN:
				helper_binary_op<BinaryOpMin<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_MAX:
				helper_binary_op<BinaryOpMax<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_EQ:
				helper_binary_op<BinaryOpCompareEq<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_NEQ:
				helper_binary_op<BinaryOpCompareNeq<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_GT:
				helper_binary_op<BinaryOpCompareGt<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_GE:
				helper_binary_op<BinaryOpCompareGe<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_LT:
				helper_binary_op<BinaryOpCompareLt<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
				break;
			case AVOCADO_BINARY_OP_COMPARE_LE:
				helper_binary_op<BinaryOpCompareLe<T>, T, U>(stream, dst, input1, alpha1, input2, alpha2, beta, dimensions);
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

		avStatus_t cudaBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			BroadcastedDimensions dimensions = getBroadcastDimensions(getTensor(aDesc), getTensor(bDesc));
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			if (is_logical(operation))
			{
				const int bytes = dimensions.last * dataTypeSize(getTensor(aDesc).dtype());
				if (bytes % 4 == 0)
					return launcher_binary_logical_op(stream, getPointer<uint32_t>(cMem), getPointer<uint32_t>(aMem), getPointer<uint32_t>(bMem), dimensions,
							operation);
				else
				{
					if (bytes % 2 == 0)
						return launcher_binary_logical_op(stream, getPointer<uint16_t>(cMem), getPointer<uint16_t>(aMem), getPointer<uint16_t>(bMem),
								dimensions, operation);
					else
						return launcher_binary_logical_op(stream, getPointer<uint8_t>(cMem), getPointer<uint8_t>(aMem), getPointer<uint8_t>(bMem), dimensions,
								operation);
				}
			}
			else
			{
				switch (getTensor(cDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT16:
						launcher_binary_op(stream, getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha1), getPointer<float16>(bMem),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_BFLOAT16:
						launcher_binary_op(stream, getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha1), getPointer<bfloat16>(bMem),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT32:
						launcher_binary_op(stream, getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha1), getPointer<float>(bMem),
								getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						launcher_binary_op(stream, getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha1), getPointer<double>(bMem),
								getAlphaValue<double>(alpha2), getBetaValue<double>(beta), dimensions, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}

			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

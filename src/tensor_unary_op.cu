/*
 * tensor_unary_op.cu
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
	class UnaryOpAbs
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::abs(x);
		}
	};
	template<typename T>
	class UnaryOpCeil
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::ceil(x);
		}
	};
	template<typename T>
	class UnaryOpCos
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::cos(x);
		}
	};
	template<typename T>
	class UnaryOpExp
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::exp(x);
		}
	};
	template<typename T>
	class UnaryOpFloor
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::floor(x);
		}
	};
	template<typename T>
	class UnaryOpLn
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::log(x);
		}
	};
	template<typename T>
	class UnaryOpNeg
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return -x;
		}
	};
	template<typename T>
	class UnaryOpRcp
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::one<T>() / x;
		}
	};
	template<typename T>
	class UnaryOpRsqrt
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::one<T>() / sqrt(x);
		}
	};
	template<typename T>
	class UnaryOpSin
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::sin(x);
		}
	};
	template<typename T>
	class UnaryOpSquare
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::square(x);
		}
	};
	template<typename T>
	class UnaryOpSqrt
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::sqrt(x);
		}
	};
	template<typename T>
	class UnaryOpTan
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return numbers::tan(x);
		}
	};
	template<typename T>
	class UnaryOpLogicalNot
	{
	public:
		__device__ numbers::Number<T> operator()(numbers::Number<T> x) const noexcept
		{
			return ~x;
		}
	};

	/* Logical operations */
	template<class Op, typename T>
	__global__ void kernel_unary_logical_op(T* dst, const T *input, const unsigned int elements)
	{
		Op operation;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> tmp(input + i, elements - i);
			tmp = operation(tmp);
			tmp.store(dst + i, elements - i);
		}
	}
	/* Arithmetic operations */
	template<class Op, typename T, typename U = T>
	__global__ void kernel_unary_op(T* dst, const T *input, const U alpha, const unsigned int elements)
	{
		numbers::Number<T> _alpha(alpha);
		Op operation;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> tmp(input + i, elements - i);
			tmp = operation(tmp * _alpha);
			tmp.store(dst + i, elements - i);
		}
	}
	template<class Op, typename T, typename U = T>
	__global__ void kernel_unary_op(T* dst, const T *input, const U alpha, const U beta, const unsigned int elements)
	{
		numbers::Number<T> _alpha(alpha);
		numbers::Number<T> _beta(beta);
		Op operation;
		for (unsigned int i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * blockDim.x * gridDim.x)
		{
			numbers::Number<T> tmp(input + i, elements - i);
			tmp = operation(tmp * _alpha);
			if (_beta != numbers::zero<T>())
				tmp += _beta * numbers::Number<T>(dst + i, elements - i);
			tmp.store(dst + i, elements - i);
		}
	}
	template<typename T, typename U = T>
	avStatus_t launcher_unary_op(cudaStream_t stream, T* dst, const T *input, const U alpha, const U beta, const unsigned int elements, avUnaryOp_t operation)
	{
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		switch (operation)
		{
			case AVOCADO_UNARY_OP_ABS:
				kernel_unary_op<UnaryOpAbs<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_CEIL:
				kernel_unary_op<UnaryOpCeil<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_COS:
				kernel_unary_op<UnaryOpCos<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_EXP:
				kernel_unary_op<UnaryOpExp<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_FLOOR:
				kernel_unary_op<UnaryOpFloor<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_LN:
				kernel_unary_op<UnaryOpLn<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_NEG:
				kernel_unary_op<UnaryOpNeg<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RCP:
				kernel_unary_op<UnaryOpRcp<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RSQRT:
				kernel_unary_op<UnaryOpRsqrt<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SIN:
				kernel_unary_op<UnaryOpSin<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQUARE:
				kernel_unary_op<UnaryOpSquare<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQRT:
				kernel_unary_op<UnaryOpSqrt<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_TAN:
				kernel_unary_op<UnaryOpTan<T>, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_LOGICAL_NOT:
				kernel_unary_logical_op<UnaryOpLogicalNot<T>, T> <<<gridDim, blockDim, 0, stream>>>(dst, input, elements);
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

		avStatus_t cudaUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			unsigned int elements = getTensor(aDesc).volume();

			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					launcher_unary_op(stream, getPointer<float16>(cMem), getPointer<float16>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							operation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					launcher_unary_op(stream, getPointer<bfloat16>(cMem), getPointer<bfloat16>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements,
							operation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					launcher_unary_op(stream, getPointer<float>(cMem), getPointer<float>(aMem), getAlphaValue(alpha), getBetaValue(beta), elements, operation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					launcher_unary_op(stream, getPointer<double>(cMem), getPointer<double>(aMem), getAlphaValue<double>(alpha), getBetaValue<double>(beta),
							elements, operation);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}

			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

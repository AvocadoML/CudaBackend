/*
 * tensor_unary_op.cu
 *
 *  Created on: Dec 26, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
#include "logical_ops.cuh"
#include "utilities.hpp"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	class UnaryOpAbs
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return abs(x);
		}
	};
	template<typename T>
	class UnaryOpCeil
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return ceil(x);
		}
	};
	template<typename T>
	class UnaryOpCos
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return cos(x);
		}
	};
	template<typename T>
	class UnaryOpExp
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return exp(x);
		}
	};
	template<typename T>
	class UnaryOpFloor
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return floor(x);
		}
	};
	template<typename T>
	class UnaryOpLn
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return log(x);
		}
	};
	template<typename T>
	class UnaryOpNeg
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return -x;
		}
	};
	template<typename T>
	class UnaryOpRcp
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return one<T>() / x;
		}
	};
	template<typename T>
	class UnaryOpRsqrt
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return one<T>() / sqrt(x);
		}
	};
	template<typename T>
	class UnaryOpSin
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return sin(x);
		}
	};
	template<typename T>
	class UnaryOpSquare
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return square(x);
		}
	};
	template<typename T>
	class UnaryOpSqrt
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return sqrt(x);
		}
	};
	template<typename T>
	class UnaryOpTan
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return tan(x);
		}
	};
	template<typename T>
	class UnaryOpLogicalNot
	{
	public:
		__device__ T operator()(T x) const noexcept
		{
			return ~x;
		}
	};

	/* Logical operations */
	template<class Op, typename T>
	__global__ void kernel_unary_logical_op(T* dst, const T *input, const unsigned int elements)
	{
		Op operation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
			dst[i] = operation(input[i]);
	}
	template<class Op, typename T>
	void helper_unary_logical_op(cudaStream_t stream, T* dst, const T *input, const unsigned int elements)
	{
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);
		kernel_unary_logical_op<Op, T> <<<gridDim, blockDim, 0, stream>>>(dst, input, elements);
	}
	template<typename T>
	avStatus_t launcher_unary_logical_op(cudaStream_t stream, T* dst, const T *input, const unsigned int elements, avUnaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_UNARY_OP_LOGICAL_NOT:
				helper_unary_logical_op<UnaryOpLogicalNot<T>, T>(stream, dst, input, elements);
				break;
			default:
				return AVOCADO_STATUS_BAD_PARAM;
		}
		return checkForErrors();
	}

	/* Arithmetic operations */
	template<class Op, typename T, typename U = T>
	__global__ void kernel_unary_op(T* dst, const T *input, const U alpha, const unsigned int elements)
	{
		Op operation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
			dst[i] = operation(static_cast<U>(input[i]) * alpha);
	}
	template<class Op, typename T, typename U = T>
	__global__ void kernel_unary_op(T* dst, const T *input, const U alpha, const U beta, const unsigned int elements)
	{
		Op operation;
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
			dst[i] = operation(static_cast<U>(input[i]) * alpha) + beta * static_cast<U>(dst[i]);
	}
	template<class Op, typename T, typename U = T>
	void helper_unary_op(cudaStream_t stream, T* dst, const T *input, const U alpha, const U beta, const unsigned int elements)
	{
		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, blockDim.x);

		if (beta == zero<T>())
			kernel_unary_op<Op, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, elements);
		else
			kernel_unary_op<Op, T, U> <<<gridDim, blockDim, 0, stream>>>(dst, input, alpha, beta, elements);
	}
	template<typename T, typename U = T>
	avStatus_t launcher_unary_op(cudaStream_t stream, T* dst, const T *input, const U alpha, const U beta, const unsigned int elements, avUnaryOp_t operation)
	{
		switch (operation)
		{
			case AVOCADO_UNARY_OP_ABS:
				helper_unary_op<UnaryOpAbs<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_CEIL:
				helper_unary_op<UnaryOpAbs<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_COS:
				helper_unary_op<UnaryOpAbs<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_EXP:
				helper_unary_op<UnaryOpExp<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_FLOOR:
				helper_unary_op<UnaryOpFloor<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_LN:
				helper_unary_op<UnaryOpLn<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_NEG:
				helper_unary_op<UnaryOpNeg<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RCP:
				helper_unary_op<UnaryOpRcp<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_RSQRT:
				helper_unary_op<UnaryOpRsqrt<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SIN:
				helper_unary_op<UnaryOpSin<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQUARE:
				helper_unary_op<UnaryOpSquare<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_SQRT:
				helper_unary_op<UnaryOpSqrt<T>, T, U>(stream, dst, input, alpha, beta, elements);
				break;
			case AVOCADO_UNARY_OP_TAN:
				helper_unary_op<UnaryOpTan<T>, T, U>(stream, dst, input, alpha, beta, elements);
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
		avStatus_t cudaUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			unsigned int elements = cuda::getTensor(aDesc).volume();

			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			if (cuda::is_logical(operation))
			{
				const int bytes = elements * cuda::dataTypeSize(cuda::getTensor(aDesc).dtype());
				if (bytes % 4 == 0)
					return launcher_unary_logical_op(stream, cuda::getPointer<uint32_t>(cMem), cuda::getPointer<uint32_t>(aMem), bytes / 4, operation);
				else
				{
					if (bytes % 2 == 0)
						return launcher_unary_logical_op(stream, cuda::getPointer<uint16_t>(cMem), cuda::getPointer<uint16_t>(aMem), bytes / 2, operation);
					else
						return launcher_unary_logical_op(stream, cuda::getPointer<uint8_t>(cMem), cuda::getPointer<uint8_t>(aMem), bytes / 1, operation);
				}
			}
			else
			{
				switch (cuda::getTensor(cDesc).dtype())
				{
//					case AVOCADO_DTYPE_FLOAT16:
//						launcher_binary_op(stream, cuda::getPointer<half>(cMem), cuda::getPointer<half>(aMem), cuda::getAlphaValue(alpha1),
//								cuda::getPointer<half>(bMem), cuda::getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
//						break;
//					case AVOCADO_DTYPE_BFLOAT16:
//						launcher_binary_op(stream, cuda::getPointer<bfloat16>(cMem), cuda::getPointer<bfloat16>(aMem), cuda::getAlphaValue(alpha1),
//								cuda::getPointer<bfloat16>(bMem), cuda::getAlphaValue(alpha2), getBetaValue(beta), dimensions, operation);
//						break;
					case AVOCADO_DTYPE_FLOAT32:
						launcher_unary_op(stream, cuda::getPointer<float>(cMem), cuda::getPointer<float>(aMem), cuda::getAlphaValue(alpha),
								cuda::getBetaValue(beta), elements, operation);
						break;
					case AVOCADO_DTYPE_FLOAT64:
						launcher_unary_op(stream, cuda::getPointer<double>(cMem), cuda::getPointer<double>(aMem), cuda::getAlphaValue<double>(alpha),
								cuda::getBetaValue<double>(beta), elements, operation);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
			}

			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

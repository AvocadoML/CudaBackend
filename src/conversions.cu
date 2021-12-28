/*
 * conversions.cu
 *
 *  Created on: Sep 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "utilities.hpp"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace avocado
{
	namespace backend
	{
		struct bfloat16
		{
			uint16_t data;
		};
	}
}

namespace
{
	using namespace avocado::backend;

	template<typename T, typename U>
	struct Converter
	{
		__device__ T convert(U x) noexcept
		{
			return static_cast<T>(x);
		}
	};

	/* half float conversions */
	template<typename U>
	struct Converter<half, U>
	{
		__device__ half convert(U x) noexcept
		{
			return half(__float2half(x));
		}
	};
	template<typename T>
	struct Converter<T, half>
	{
		__device__ T convert(half x) noexcept
		{
			return static_cast<T>(__half2float(x));
		}
	};
	template<>
	struct Converter<half, half>
	{
		__device__ half convert(half x) noexcept
		{
			return x;
		}
	};
	template<>
	struct Converter<half, float2>
	{
		__device__ half convert(float2 x) noexcept
		{
			return Converter<half, float>().convert(x.x);
		}
	};
	template<>
	struct Converter<half, double2>
	{
		__device__ half convert(double2 x) noexcept
		{
			return Converter<half, double>().convert(x.x);
		}
	};
	template<>
	struct Converter<float2, half>
	{
		__device__ float2 convert(half x) noexcept
		{
			return
			{	Converter<float, half>().convert(x), 0.0f};
		}
	};
	template<>
	struct Converter<double2, half>
	{
		__device__ double2 convert(half x) noexcept
		{
			return
			{	Converter<double, half>().convert(x), 0.0};
		}
	};

	/* bfloat conversions */
	template<typename U>
	struct Converter<bfloat16, U>
	{
		__device__ bfloat16 convert(U x) noexcept
		{
			float tmp = static_cast<float>(x);
			return reinterpret_cast<bfloat16*>(&tmp)[0];
		}
	};
	template<typename T>
	struct Converter<T, bfloat16>
	{
		__device__ T convert(bfloat16 x) noexcept
		{
			float tmp = 0.0f;
			reinterpret_cast<bfloat16*>(&tmp)[0] = x;
			return static_cast<T>(tmp);
		}
	};
	template<>
	struct Converter<bfloat16, bfloat16>
	{
		__device__ bfloat16 convert(bfloat16 x) noexcept
		{
			return x;
		}
	};
	template<>
	struct Converter<bfloat16, float2>
	{
		__device__ bfloat16 convert(float2 x) noexcept
		{
			return Converter<bfloat16, float>().convert(x.x);
		}
	};
	template<>
	struct Converter<bfloat16, double2>
	{
		__device__ bfloat16 convert(double2 x) noexcept
		{
			return Converter<bfloat16, double>().convert(x.x);
		}
	};
	template<>
	struct Converter<float2, bfloat16>
	{
		__device__ float2 convert(bfloat16 x) noexcept
		{
			return
			{	Converter<float, bfloat16>().convert(x), 0.0f};
		}
	};
	template<>
	struct Converter<double2, bfloat16>
	{
		__device__ double2 convert(bfloat16 x) noexcept
		{
			return
			{	Converter<double, bfloat16>().convert(x), 0.0};
		}
	};

	template<>
	struct Converter<half, bfloat16>
	{
		__device__ half convert(bfloat16 x) noexcept
		{
			return __float2half(Converter<float, bfloat16>().convert(x));
		}
	};
	template<>
	struct Converter<bfloat16, half>
	{
		__device__ bfloat16 convert(half x) noexcept
		{
			return Converter<bfloat16, float>().convert(__half2float(x));
		}
	};

	/* complex types conversion */
	template<typename T>
	struct Converter<T, float2>
	{
		__device__ T convert(float2 x) noexcept
		{
			return Converter<T, float>().convert(x.x);
		}
	};
	template<typename T>
	struct Converter<T, double2>
	{
		__device__ T convert(double2 x) noexcept
		{
			return Converter<T, double>().convert(x.x);
		}
	};
	template<typename U>
	struct Converter<float2, U>
	{
		__device__ float2 convert(U x) noexcept
		{
			return
			{	static_cast<float>(x), 0.0f};
		}
	};
	template<typename U>
	struct Converter<double2, U>
	{
		__device__ double2 convert(U x) noexcept
		{
			return
			{	static_cast<double>(x), 0.0};
		}
	};

	template<>
	struct Converter<float2, float2>
	{
		__device__ float2 convert(float2 x) noexcept
		{
			return x;
		}
	};
	template<>
	struct Converter<float2, double2>
	{
		__device__ float2 convert(double2 x) noexcept
		{
			return
			{	static_cast<float>(x.x), static_cast<float>(x.y)};
		}
	};
	template<>
	struct Converter<double2, float2>
	{
		__device__ double2 convert(float2 x) noexcept
		{
			return
			{	static_cast<double>(x.x), static_cast<double>(x.y)};
		}
	};
	template<>
	struct Converter<double2, double2>
	{
		__device__ double2 convert(double2 x) noexcept
		{
			return x;
		}
	};

	template<typename T, typename U>
	__global__ void kernel_convert(T *dst, const U *src, avSize_t elements)
	{
		for (avSize_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] = Converter<T, U>().convert(src[i]);
	}
	template<typename T>
	void convert_helper(cudaStream_t stream, T *dst, const void *src, avDataType_t srcType, avSize_t elements)
	{
		assert(dst != nullptr);
		assert(src != nullptr);

		dim3 blockDim(256);
		dim3 gridDim = gridSize<1024>(elements, 256);

		switch (srcType)
		{
			case AVOCADO_DTYPE_UINT8:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const uint8_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT8:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const int8_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT16:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const int16_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT32:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const int32_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_INT64:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const int64_t*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT16:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const half*>(src), elements);
				break;
			case AVOCADO_DTYPE_BFLOAT16:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const bfloat16*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT32:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const float*>(src), elements);
				break;
			case AVOCADO_DTYPE_FLOAT64:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const double*>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX32:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const float2*>(src), elements);
				break;
			case AVOCADO_DTYPE_COMPLEX64:
				kernel_convert<<<gridDim, blockDim, 0, stream>>>(dst, reinterpret_cast<const double2*>(src), elements);
				break;
			default:
				break;
		}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType, const avMemoryDescriptor_t src,
				avDataType_t srcType, avSize_t elements)
		{
			switch (dstType)
			{
				case AVOCADO_DTYPE_UINT8:
					convert_helper(getContext(context).getStream(), getPointer<uint8_t>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_INT8:
					convert_helper(getContext(context).getStream(), getPointer<int8_t>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_INT16:
					convert_helper(getContext(context).getStream(), getPointer<int16_t>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_INT32:
					convert_helper(getContext(context).getStream(), getPointer<int32_t>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_INT64:
					convert_helper(getContext(context).getStream(), getPointer<int64_t>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					convert_helper(getContext(context).getStream(), getPointer<half>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					convert_helper(getContext(context).getStream(), getPointer<bfloat16>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					convert_helper(getContext(context).getStream(), getPointer<float>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					convert_helper(getContext(context).getStream(), getPointer<double>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
					convert_helper(getContext(context).getStream(), getPointer<float2>(dst), getPointer(src), srcType, elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
					convert_helper(getContext(context).getStream(), getPointer<double2>(dst), getPointer(src), srcType, elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	}
/* namespace backend */
} /* namespace avocado */

/*
 * bf16_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef BF16_NUMBER_CUH_
#define BF16_NUMBER_CUH_

#include "generic_number.cuh"

#if __has_include(<cuda_bf16.h>)
#  include <cuda_bf16.hpp>
#  define HAS_BF16_HEADER 1
#else
#  define HAS_BF16_HEADER 0
#endif

#define BF16_COMPUTE_MIN_ARCH 800

#if HAS_BF16_HEADER
typedef __nv_bfloat16 bfloat16;
typedef __nv_bfloat162 bfloat16x2;
#else
namespace avocado
{
	namespace backend
	{
		struct bfloat16
		{
			uint16_t x;
		};
	} /* namespace backend */
} /* namespace avocado */
#endif

namespace numbers
{
	using avocado::backend::bfloat16;

#if (__CUDA_ARCH__ < BF16_COMPUTE_MIN_ARCH) or not HAS_BF16_HEADER
	__device__ __host__ bfloat16 float_to_bfloat16(float x)
	{
		return reinterpret_cast<bfloat16*>(&x)[1];
	}
	__device__ __host__ float bfloat16_to_float(bfloat16 x) noexcept
	{
		float result = 0.0f;
		reinterpret_cast<bfloat16*>(&result)[1] = x;
		return result;
	}
#endif

	template<>
	class Number<bfloat16>
	{
	private:
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// bfloat16x2 m_data;
#else
		float m_data = 0.0f;
#endif
	public:
		__device__ Number() = default;
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
//		__device__ Number(bfloat162 x) :
//		m_data(x)
//		{
//		}
#else
		__device__ Number(bfloat16 x) :
				m_data(bfloat16_to_float(x))
		{
		}
		__device__ Number(float x) :
				m_data(x)
		{
		}
#endif
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
//		__device__ Number(const bfloat16 *ptr, int num = 2)
//		{
//			load(ptr, num);
//		}
//		__device__ Number(const float *ptr, int num = 2)
//		{
//			load(ptr, num);
//		}
//		__device__ void load(const bfloat16 *ptr, int num = 2)
//		{
//			assert(ptr != nullptr);
//			if (num >= 2)
//			m_data = reinterpret_cast<const bfloat162*>(ptr)[0];
//			else
//			{
//				if (num == 1)
//				m_data = bfloat162(ptr[0], 0.0f);
//				else
//				m_data = bfloat162(0.0f, 0.0f);
//			}
//		}
//		__device__ void load(const float *ptr, int num = 2)
//		{
//			assert(ptr != nullptr);
//			if (num >= 2)
//			m_data = bfloat162(ptr[0], ptr[1]);
//			else
//			{
//				if (num == 1)
//				m_data = bfloat162(ptr[0], 0.0f);
//				else
//				m_data = bfloat162(0.0f, 0.0f);
//			}
//		}
//		__device__ void store(bfloat16 *ptr, int num = 2) const
//		{
//			assert(ptr != nullptr);
//			switch (num)
//			{
//				default:
//				case 2:
//				reinterpret_cast<bfloat162*>(ptr)[0] = m_data;
//				break;
//				case 1:
//				ptr[0] = m_data.x;
//				break;
//			}
//		}
//		__device__ void store(float *ptr, int num = 2) const
//		{
//			assert(ptr != nullptr);
//			switch (num)
//			{
//				default:
//				case 2:
//				ptr[0] = static_cast<float>(m_data.x);
//				ptr[1] = static_cast<float>(m_data.y);
//				break;
//				case 1:
//				ptr[0] = m_data.x;
//				break;
//			}
//		}
//		__device__ operator bfloat162() const
//		{
//			return m_data;
//		}
#else
		__device__ Number(const bfloat16 *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ Number(const float *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ void load(const bfloat16 *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = bfloat16_to_float(ptr[0]);
		}
		__device__ void load(const float *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = ptr[0];
		}
		__device__ void store(bfloat16 *ptr, int num = 1) const
		{
			assert(ptr != nullptr);
			if (num >= 1)
				ptr[0] = float_to_bfloat16(m_data);
		}
		__device__ void store(float *ptr, int num = 1) const
		{
			assert(ptr != nullptr);
			if (num >= 1)
				ptr[0] = m_data;
		}
		__device__ operator float() const
		{
			return m_data;
		}
#endif
	};

	template<>
	__device__ int length<bfloat16>()
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		return 2;
#else
		return 1;
#endif
	}

	template<>
	__device__ Number<bfloat16> zero()
	{
		return Number<bfloat16>(0.0f);
	}
	template<>
	__device__ Number<bfloat16> one()
	{
		return Number<bfloat16>(1.0f);
	}
	template<>
	__device__ Number<bfloat16> epsilon()
	{
		return Number<bfloat16>(1.1920928955078125e-7f);
	}

	__device__ Number<bfloat16> pow(Number<bfloat16> x, Number<bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return powf(x, y);
#endif
	}
	__device__ Number<bfloat16> mod(Number<bfloat16> x, Number<bfloat16> y)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return fmodf(x, y);
#endif
	}
	__device__ Number<bfloat16> exp(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return expf(x);
#endif
	}
	__device__ Number<bfloat16> log(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return logf(x);
#endif
	}
	__device__ Number<bfloat16> tanh(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return tanhf(x);
#endif
	}
	__device__ Number<bfloat16> expm1(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return expm1f(x);
#endif
	}
	__device__ Number<bfloat16> log1p(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return log1pf(x);
#endif
	}
	__device__ Number<bfloat16> sin(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return sinf(x);
#endif
	}
	__device__ Number<bfloat16> cos(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return cosf(x);
#endif
	}
	__device__ Number<bfloat16> tan(Number<bfloat16> x)
	{
#if (__CUDA_ARCH__ >= BF16_COMPUTE_MIN_ARCH) and HAS_BF16_HEADER
		// TODO
#else
		return tanf(x);
#endif
	}

} /* namespace numbers */

#endif /* BF16_NUMBER_CUH_ */

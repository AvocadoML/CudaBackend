/*
 * fp16_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef FP16_NUMBER_CUH_
#define FP16_NUMBER_CUH_

#include "generic_number.cuh"

#include <cuda_fp16.hpp>

#define FP16_COMPUTE_MIN_ARCH 700
#define FP16_STORAGE_MIN_ARCH 530

namespace avocado
{
	namespace backend
	{
		typedef half float16;
		typedef half2 float16x2;
	} /* namespace backend */
} /* namespace avocado */

namespace numbers
{
	using avocado::backend::float16;
	using avocado::backend::float16x2;

	template<>
	class Number<float16>
	{
	private:
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		float16x2 m_data;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		float m_data = 0.0f;
#else
#endif
	public:
		__device__ Number() = default;
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		__device__ Number(float16x2 x) :
		m_data(x)
		{
		}
		__device__ Number(float16 x) :
		m_data(x, x)
		{
		}
		__device__ Number(float16 x, float16 y) :
		m_data(x, y)
		{
		}
		__device__ Number(float x) :
		m_data(x, x)
		{
		}
		__device__ Number(float x, float y) :
		m_data(x, y)
		{
		}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		__device__ Number(float16 x) :
				m_data(x)
		{
		}
		__device__ Number(float x) :
				m_data(x)
		{
		}
#else
		__device__ Number(float16 x)
		{
		}
		__device__ Number(float x)
		{
		}
		__device__ Number(double x)
		{
		}
#endif
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		__device__ Number(const float16 *ptr, int num = 2)
		{
			load(ptr, num);
		}
		__device__ Number(const float *ptr, int num = 2)
		{
			load(ptr, num);
		}
		__device__ void load(const float16 *ptr, int num = 2)
		{
			assert(ptr != nullptr);
			if (num >= 2)
			m_data = reinterpret_cast<const float16x2*>(ptr)[0];
			else
			{
				if (num == 1)
				m_data = float16x2(ptr[0], 0.0f);
				else
				m_data = float16x2(0.0f, 0.0f);
			}
		}
		__device__ void load(const float *ptr, int num = 2)
		{
			assert(ptr != nullptr);
			if (num >= 2)
			m_data = float16x2(ptr[0], ptr[1]);
			else
			{
				if (num == 1)
				m_data = float16x2(ptr[0], 0.0f);
				else
				m_data = float16x2(0.0f, 0.0f);
			}
		}
		__device__ void store(float16 *ptr, int num = 2) const
		{
			assert(ptr != nullptr);
			switch (num)
			{
				default:
				case 2:
				reinterpret_cast<float16x2*>(ptr)[0] = m_data;
				break;
				case 1:
				ptr[0] = m_data.x;
				break;
			}
		}
		__device__ void store(float *ptr, int num = 2) const
		{
			assert(ptr != nullptr);
			switch (num)
			{
				default:
				case 2:
				ptr[0] = static_cast<float>(m_data.x);
				ptr[1] = static_cast<float>(m_data.y);
				break;
				case 1:
				ptr[0] = m_data.x;
				break;
			}
		}
		__device__ operator float16x2() const
		{
			return m_data;
		}
		__device__ float16 low() const
		{
			return m_data.x;
		}
		__device__ float16 high() const
		{
			return m_data.y;
		}
		__device__ Number<float16> operator-() const
		{
			return Number<float16>(-m_data);
		}
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		__device__ Number(const float16 *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ Number(const float *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ void load(const float16 *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = static_cast<float>(ptr[0]);
		}
		__device__ void load(const float *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = ptr[0];
		}
		__device__ void store(float16 *ptr, int num = 1) const
		{
			assert(ptr != nullptr);
			if (num >= 1)
				ptr[0] = m_data;
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
		__device__ Number<float16> operator-() const
		{
			return Number<float16>(-m_data);
		}
#else
		__device__ Number(const float16 *ptr, int num = 0)
		{
		}
		__device__ Number(const float *ptr, int num = 0)
		{
		}
		__device__ void load(const float16 *ptr, int num = 0)
		{
		}
		__device__ void load(const float *ptr, int num = 0)
		{
		}
		__device__ void store(float16 *ptr, int num = 0) const
		{
		}
		__device__ void store(float *ptr, int num = 0) const
		{
		}
		__device__ operator float() const
		{
			return 0.0f;
		}
		__device__ Number<float16> operator-() const
		{
			return Number<float16>(0.0f);
		}
#endif
	};

	template<>
	DEVICE_INLINE int length<float16>()
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return 2;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return 1;
#else
		return 0;
#endif
	}

	template<>
	DEVICE_INLINE Number<float16> zero()
	{
		return Number<float16>(0.0f);
	}
	template<>
	DEVICE_INLINE Number<float16> one()
	{
		return Number<float16>(1.0f);
	}
	template<>
	DEVICE_INLINE Number<float16> epsilon()
	{
		return Number<float16>(0.00006103515625f);
	}

	DEVICE_INLINE Number<float16> sgn(Number<float16> x) noexcept
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		float16x2 tmp = x;
		float16x2 result;
		result.x = static_cast<float16>(internal::sgn(static_cast<float>(tmp.x)));
		result.y = static_cast<float16>(internal::sgn(static_cast<float>(tmp.y)));
		return result;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Number<float16>(internal::sgn(static_cast<float>(x)));
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> abs(Number<float16> x) noexcept
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		float16x2 tmp = x;
		float16x2 result;
		result.x = static_cast<float16>(fabsf(static_cast<float>(tmp.x)));
		result.y = static_cast<float16>(fabsf(static_cast<float>(tmp.y)));
		return result;
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return Number<float16>(fabsf(static_cast<float>(x)));
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> max(Number<float16> x, Number<float16> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(fmax(static_cast<float>(x.low()), static_cast<float>(y.low())), fmax(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmax(x, y);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> min(Number<float16> x, Number<float16> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(fmin(static_cast<float>(x.low()), static_cast<float>(y.low())), fmin(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmin(x, y);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> pow(Number<float16> x, Number<float16> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(powf(static_cast<float>(x.low()), static_cast<float>(y.low())), powf(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return powf(x, y);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> mod(Number<float16> x, Number<float16> y)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(fmodf(static_cast<float>(x.low()), static_cast<float>(y.low())),
				fmodf(static_cast<float>(x.high()), static_cast<float>(y.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return fmodf(x, y);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> exp(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> log(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return logf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> tanh(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(tanhf(static_cast<float>(x.low())), tanhf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanhf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> expm1(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2exp(x) - one<float16>();
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return expm1f(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> log1p(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2log(one<float16>() + x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return log1pf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> sin(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2sin(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return sinf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> cos(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return h2cos(x);
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return cosf(x);
#else
		return Number<float16>();
#endif
	}
	DEVICE_INLINE Number<float16> tan(Number<float16> x)
	{
#if __CUDA_ARCH__ >= FP16_COMPUTE_MIN_ARCH
		return Number<float16>(tanf(static_cast<float>(x.low())), tanf(static_cast<float>(x.high())));
#elif __CUDA_ARCH__ >= FP16_STORAGE_MIN_ARCH
		return tanf(x);
#else
		return Number<float16>();
#endif
	}

} /* namespace numbers */

#endif /* FP16_NUMBER_CUH_ */

/*
 * fp32_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef FP32_NUMBER_CUH_
#define FP32_NUMBER_CUH_

#include "generic_number.cuh"

namespace numbers
{
	template<>
	class Number<float>
	{
	private:
		float m_data;
	public:
		__device__ Number() = default;
		__device__ Number(float x) :
				m_data(x)
		{
		}
		__device__ Number(double x) :
				m_data(static_cast<float>(x))
		{
		}
		__device__ Number(const float *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ void load(const float *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = ptr[0];
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
		__device__ Number<float> operator-() const
		{
			return Number<float>(-m_data);
		}
		__device__ Number<float> operator~() const
		{
			const int32_t tmp = ~reinterpret_cast<const int32_t*>(&m_data)[0];
			return Number<float>(reinterpret_cast<const float*>(&tmp)[0]);
		}
	};

	template<>
	DEVICE_INLINE Number<float> zero()
	{
		return Number<float>(0.0f);
	}
	template<>
	DEVICE_INLINE Number<float> one()
	{
		return Number<float>(1.0f);
	}
	template<>
	DEVICE_INLINE Number<float> epsilon()
	{
		return Number<float>(1.1920928955078125e-7f);
	}

	DEVICE_INLINE Number<float> operator+(const Number<float> &lhs, const Number<float> &rhs)
	{
		return Number<float>(static_cast<float>(lhs) + static_cast<float>(rhs));
	}
	DEVICE_INLINE Number<float> operator-(const Number<float> &lhs, const Number<float> &rhs)
	{
		return Number<float>(static_cast<float>(lhs) - static_cast<float>(rhs));
	}
	DEVICE_INLINE Number<float> operator*(const Number<float> &lhs, const Number<float> &rhs)
	{
		return Number<float>(static_cast<float>(lhs) * static_cast<float>(rhs));
	}
	DEVICE_INLINE Number<float> operator/(const Number<float> &lhs, const Number<float> &rhs)
	{
		return Number<float>(static_cast<float>(lhs) / static_cast<float>(rhs));
	}

	DEVICE_INLINE Number<float> sgn(Number<float> x) noexcept
	{
		return internal::sgn(static_cast<float>(x));
	}
	DEVICE_INLINE Number<float> abs(Number<float> x) noexcept
	{
		return fabsf(x);
	}
	DEVICE_INLINE Number<float> max(Number<float> x, Number<float> y)
	{
		return fmax(x, y);
	}
	DEVICE_INLINE Number<float> min(Number<float> x, Number<float> y)
	{
		return fmin(x, y);
	}
	DEVICE_INLINE Number<float> ceil(Number<float> x)
	{
		return ceilf(x);
	}
	DEVICE_INLINE Number<float> floor(Number<float> x)
	{
		return floorf(x);
	}
	DEVICE_INLINE Number<float> sqrt(Number<float> x)
	{
		return sqrtf(x);
	}
	DEVICE_INLINE Number<float> pow(Number<float> x, Number<float> y)
	{
		return powf(x, y);
	}
	DEVICE_INLINE Number<float> mod(Number<float> x, Number<float> y)
	{
		return fmodf(x, y);
	}
	DEVICE_INLINE Number<float> exp(Number<float> x)
	{
		return expf(x);
	}
	DEVICE_INLINE Number<float> log(Number<float> x)
	{
		return logf(x);
	}
	DEVICE_INLINE Number<float> tanh(Number<float> x)
	{
		return tanhf(x);
	}
	DEVICE_INLINE Number<float> expm1(Number<float> x)
	{
		return expm1f(x);
	}
	DEVICE_INLINE Number<float> log1p(Number<float> x)
	{
		return log1pf(x);
	}
	DEVICE_INLINE Number<float> sin(Number<float> x)
	{
		return sinf(x);
	}
	DEVICE_INLINE Number<float> cos(Number<float> x)
	{
		return cosf(x);
	}
	DEVICE_INLINE Number<float> tan(Number<float> x)
	{
		return tanf(x);
	}

} /* namespace numbers */

#endif /* FP32_NUMBER_CUH_ */

/*
 * fp64_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef FP64_NUMBER_CUH_
#define FP64_NUMBER_CUH_

#include "generic_number.cuh"

namespace numbers
{
	template<>
	class Number<double>
	{
	private:
		double m_data;
	public:
		__device__ Number() = default;
		__device__ Number(double x) :
				m_data(x)
		{
		}
		__device__ Number(float x) :
				m_data(static_cast<double>(x))
		{
		}
		__device__ Number(const double *ptr, int num = 1)
		{
			load(ptr, num);
		}
		__device__ void load(const double *ptr, int num = 1)
		{
			assert(ptr != nullptr);
			if (num >= 1)
				m_data = ptr[0];
		}
		__device__ void store(double *ptr, int num = 1) const
		{
			assert(ptr != nullptr);
			if (num >= 1)
				ptr[0] = m_data;
		}
		__device__ operator double() const
		{
			return m_data;
		}
		__device__ Number<double> operator-() const
		{
			return Number<double>(-m_data);
		}
		__device__ Number<double> operator~() const
		{
			const int64_t tmp = ~reinterpret_cast<const int64_t*>(&m_data)[0];
			return Number<double>(reinterpret_cast<const double*>(&tmp)[0]);
		}
	};

	template<>
	DEVICE_INLINE Number<double> zero()
	{
		return Number<double>(0.0);
	}
	template<>
	DEVICE_INLINE Number<double> one()
	{
		return Number<double>(1.0);
	}
	template<>
	DEVICE_INLINE Number<double> epsilon()
	{
		return Number<double>(2.22044604925031308084726333618164062e-16);
	}

	DEVICE_INLINE Number<double> operator+(const Number<double> &lhs, const Number<double> &rhs)
	{
		return Number<double>(static_cast<double>(lhs) + static_cast<double>(rhs));
	}
	DEVICE_INLINE Number<double> operator-(const Number<double> &lhs, const Number<double> &rhs)
	{
		return Number<double>(static_cast<double>(lhs) - static_cast<double>(rhs));
	}
	DEVICE_INLINE Number<double> operator*(const Number<double> &lhs, const Number<double> &rhs)
	{
		return Number<double>(static_cast<double>(lhs) * static_cast<double>(rhs));
	}
	DEVICE_INLINE Number<double> operator/(const Number<double> &lhs, const Number<double> &rhs)
	{
		return Number<double>(static_cast<double>(lhs) / static_cast<double>(rhs));
	}

	DEVICE_INLINE Number<double> sgn(Number<double> x) noexcept
	{
		return Number<double>(internal::sgn(static_cast<double>(x)));
	}
	DEVICE_INLINE Number<double> abs(Number<double> x)
	{
		return fabs(x);
	}
	DEVICE_INLINE Number<double> max(Number<double> x, Number<double> y)
	{
		return fmax(x, y);
	}
	DEVICE_INLINE Number<double> min(Number<double> x, Number<double> y)
	{
		return fmin(x, y);
	}
	DEVICE_INLINE Number<double> ceil(Number<double> x)
	{
		return ceilf(x);
	}
	DEVICE_INLINE Number<double> floor(Number<double> x)
	{
		return floorf(x);
	}
	DEVICE_INLINE Number<double> sqrt(Number<double> x)
	{
		return sqrtf(x);
	}
	DEVICE_INLINE Number<double> pow(Number<double> x, Number<double> y)
	{
		return powf(x, y);
	}
	DEVICE_INLINE Number<double> mod(Number<double> x, Number<double> y)
	{
		return fmodf(x, y);
	}
	DEVICE_INLINE Number<double> exp(Number<double> x)
	{
		return expf(x);
	}
	DEVICE_INLINE Number<double> log(Number<double> x)
	{
		return logf(x);
	}
	DEVICE_INLINE Number<double> tanh(Number<double> x)
	{
		return tanhf(x);
	}
	DEVICE_INLINE Number<double> expm1(Number<double> x)
	{
		return expm1f(x);
	}
	DEVICE_INLINE Number<double> log1p(Number<double> x)
	{
		return log1pf(x);
	}
	DEVICE_INLINE Number<double> sin(Number<double> x)
	{
		return sinf(x);
	}
	DEVICE_INLINE Number<double> cos(Number<double> x)
	{
		return cosf(x);
	}
	DEVICE_INLINE Number<double> tan(Number<double> x)
	{
		return tanf(x);
	}

	DEVICE_INLINE double horizontal_add(Number<double> x)
	{
		return static_cast<double>(x);
	}
	DEVICE_INLINE double horizontal_mul(Number<double> x)
	{
		return static_cast<double>(x);
	}
	DEVICE_INLINE double horizontal_max(Number<double> x)
	{
		return static_cast<double>(x);
	}
	DEVICE_INLINE double horizontal_min(Number<double> x)
	{
		return static_cast<double>(x);
	}
	DEVICE_INLINE double horizontal_or(Number<double> x)
	{
		return static_cast<double>(x);
	}
	DEVICE_INLINE double horizontal_and(Number<double> x)
	{
		return static_cast<double>(x);
	}

} /* namespace numbers */

#endif /* FP64_NUMBER_CUH_ */

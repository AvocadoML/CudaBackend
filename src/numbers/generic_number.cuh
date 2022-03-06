/*
 * generic_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef GENERIC_NUMBER_CUH_
#define GENERIC_NUMBER_CUH_

namespace internal
{
	template<typename T>
	__device__ T sgn(T x) noexcept
	{
		return (static_cast<T>(0.0) < x) - (x < static_cast<T>(0.0));
	}
}

namespace numbers
{
#define DEVICE_INLINE __device__ __forceinline__

	template<typename T, class dummy = T>
	class Number;

	template<typename T>
	__device__ bool is_aligned(const void * ptr)
	{
		return (reinterpret_cast<std::uintptr_t>(ptr) % sizeof(T)) == 0;
	}

	template<typename T>
	__device__ constexpr int length()
	{
		return 1;
	}

	template<typename T>
	__device__ Number<T> zero()
	{
		return Number<T>();
	}
	template<typename T>
	__device__ Number<T> one()
	{
		return Number<T>();
	}
	template<typename T>
	__device__ Number<T> epsilon()
	{
		return Number<T>();
	}

	template<typename T>
	__device__ Number<T> square(Number<T> x)
	{
		return x * x;
	}
	template<typename T>
	__device__ Number<T> sgn(Number<T> x) noexcept
	{
		if (x > zero<T>())
			return one<T>();
		else
		{
			if (x < zero<T>())
				return -one<T>();
			else
				return zero<T>();
		}
	}

	template<typename T>
	__device__ Number<T> operator+(const Number<T> &lhs, T rhs)
	{
		return lhs + Number<T>(rhs);
	}
	template<typename T>
	__device__ Number<T> operator+(T lhs, const Number<T> &rhs)
	{
		return Number<T>(lhs) + rhs;
	}
	template<typename T>
	__device__ Number<T>& operator+=(Number<T> &lhs, const Number<T> &rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}
	template<typename T>
	__device__ Number<T>& operator+=(Number<T> &lhs, T rhs)
	{
		lhs = lhs + rhs;
		return lhs;
	}

	template<typename T>
	__device__ Number<T> operator-(const Number<T> &lhs, T rhs)
	{
		return lhs - Number<T>(rhs);
	}
	template<typename T>
	__device__ Number<T> operator-(T lhs, const Number<T> &rhs)
	{
		return Number<T>(lhs) - rhs;
	}
	template<typename T>
	__device__ Number<T>& operator-=(Number<T> &lhs, const Number<T> &rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}
	template<typename T>
	__device__ Number<T>& operator-=(Number<T> &lhs, T rhs)
	{
		lhs = lhs - rhs;
		return lhs;
	}

	template<typename T>
	__device__ Number<T> operator*(const Number<T> &lhs, T rhs)
	{
		return lhs * Number<T>(rhs);
	}
	template<typename T>
	__device__ Number<T> operator*(T lhs, const Number<T> &rhs)
	{
		return Number<T>(lhs) * rhs;
	}
	template<typename T>
	__device__ Number<T>& operator*=(Number<T> &lhs, const Number<T> &rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}
	template<typename T>
	__device__ Number<T>& operator*=(Number<T> &lhs, T rhs)
	{
		lhs = lhs * rhs;
		return lhs;
	}

	template<typename T>
	__device__ Number<T> operator/(const Number<T> &lhs, T rhs)
	{
		return lhs / Number<T>(rhs);
	}
	template<typename T>
	__device__ Number<T> operator/(T lhs, const Number<T> &rhs)
	{
		return Number<T>(lhs) / rhs;
	}
	template<typename T>
	__device__ Number<T>& operator/=(Number<T> &lhs, const Number<T> &rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}
	template<typename T>
	__device__ Number<T>& operator/=(Number<T> &lhs, T rhs)
	{
		lhs = lhs / rhs;
		return lhs;
	}

} /* namespace numbers */

#endif /* GENERIC_NUMBER_CUH_ */

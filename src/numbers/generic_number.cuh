/*
 * generic_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef GENERIC_NUMBER_CUH_
#define GENERIC_NUMBER_CUH_

#define DEVICE_INLINE __device__ __forceinline__

#if __has_include(<cuda_bf16.h>)
#  define HAS_BF16_HEADER 1
#else
#  define HAS_BF16_HEADER 0
#endif

#define BF16_COMPUTE_MIN_ARCH 800

#define FP16_COMPUTE_MIN_ARCH 700
#define FP16_STORAGE_MIN_ARCH 530

namespace avocado
{
	namespace backend
	{
#if HAS_BF16_HEADER
		typedef __nv_bfloat16 bfloat16;
		typedef __nv_bfloat162 bfloat16x2;
#else
		struct bfloat16
		{
			uint16_t x;
		};
#endif

		typedef half float16;
		typedef half2 float16x2;
	} /* namespace backend */
} /* namespace avocado */

namespace internal
{
	template<typename T>
	__device__ T sgn(T x) noexcept
	{
		return (static_cast<T>(0.0) < x) - (x < static_cast<T>(0.0));
	}
} /* namespace internal */

namespace numbers
{
#if (__CUDA_ARCH__ < BF16_COMPUTE_MIN_ARCH) or not HAS_BF16_HEADER
	DEVICE_INLINE __host__ avocado::backend::bfloat16 float_to_bfloat16(float x)
	{
		return reinterpret_cast<avocado::backend::bfloat16*>(&x)[1];
	}
	DEVICE_INLINE __host__ float bfloat16_to_float(avocado::backend::bfloat16 x) noexcept
	{
		float result = 0.0f;
		reinterpret_cast<avocado::backend::bfloat16*>(&result)[1] = x;
		return result;
	}
#endif

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

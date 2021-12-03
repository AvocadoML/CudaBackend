/*
 * activations.cuh
 *
 *  Created on: Nov 23, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef ACTIVATIONS_CUH_
#define ACTIVATIONS_CUH_

template<typename T>
__host__ __device__ T zero() noexcept
{
	return static_cast<T>(0);
}
template<typename T>
__host__ __device__ T one() noexcept
{
	return static_cast<T>(1);
}
template<typename T>
__host__ __device__ T eps() noexcept
{
	return static_cast<T>(1.0e-16);
}

template<typename T>
__host__ __device__ T square(T x) noexcept
{
	return x * x;
}
template<typename T>
__host__ __device__ T safe_log(T x) noexcept
{
	return std::log(eps<T>() + x);
}
template<typename T>
__host__ __device__ bool ispow2(T x) noexcept
{
	return x > zero<T>() && !(x & (x - one<T>()));
}
template<typename T>
__host__ __device__ T sgn(T x) noexcept
{
	return (zero<T>() < x) - (x < zero<T>());
}

#endif /* ACTIVATIONS_CUH_ */

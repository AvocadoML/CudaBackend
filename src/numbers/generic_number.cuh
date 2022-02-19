/*
 * generic_number.cuh
 *
 *  Created on: Feb 18, 2022
 *      Author: Maciej Kozarzewski
 */

#ifndef GENERIC_NUMBER_CUH_
#define GENERIC_NUMBER_CUH_

namespace numbers
{

	template<typename T> // , class dummy = T>
	class Number;

	template<typename T>
	__device__ int length()
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


} /* namespace numbers */

#endif /* GENERIC_NUMBER_CUH_ */

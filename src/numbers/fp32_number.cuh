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
		float m_data = 0.0f;
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
	};

	template<>
	__device__ Number<float> zero()
	{
		return Number<float>(0.0f);
	}
	template<>
	__device__ Number<float> one()
	{
		return Number<float>(1.0f);
	}
	template<>
	__device__ Number<float> epsilon()
	{
		return Number<float>(1.1920928955078125e-7f);
	}

} /* namespace numbers */

#endif /* FP32_NUMBER_CUH_ */

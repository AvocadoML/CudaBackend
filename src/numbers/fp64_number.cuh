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
		double m_data = 0.0;
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
	};

	template<>
	__device__ Number<double> zero()
	{
		return Number<double>(0.0);
	}
	template<>
	__device__ Number<double> one()
	{
		return Number<double>(1.0);
	}
	template<>
	__device__ Number<double> epsilon()
	{
		return Number<double>(2.22044604925031308084726333618164062e-16);
	}

} /* namespace numbers */

#endif /* FP64_NUMBER_CUH_ */

/*
 * test_batchnorm.cpp
 *
 *  Created on: Jan 24, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace avocado
{
	namespace backend
	{
		TEST(TestBatchNorm, float32)
		{
			BatchNormTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f;
			float beta = 0.1f;
			float alpha2 = 1.2f;
			float beta2 = 0.2f;
			EXPECT_LT(data.getDifferenceInference(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta, &alpha2, &beta2), 1.0e-3);
		}
		TEST(TestBatchNorm, float64)
		{
			BatchNormTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1;
			double beta = 0.1;
			double alpha2 = 1.2;
			double beta2 = 0.2;
			EXPECT_LT(data.getDifferenceInference(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta, &alpha2, &beta2), 1.0e-4);
		}

	} /* namespace backend */
} /* namespace avocado */


/*
 * test_losses.cpp
 *
 *  Created on: Jan 26, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestLosses: public testing::TestWithParam<avocado::backend::avLossType_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestLosses, float32_loss)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			LossFunctionTester data(GetParam(), { 203, 405 }, AVOCADO_DTYPE_FLOAT32);
			EXPECT_LT(data.getDifferenceLoss(), 1.0e-3);
		}
		TEST_P(TestLosses, float32_gradient)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			LossFunctionTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceGradient(&alpha, &beta, true), 1.0e-3);
			EXPECT_LT(data.getDifferenceGradient(&alpha, &beta, false), 1.0e-3);
		}

		TEST_P(TestLosses, float64_loss)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			LossFunctionTester data(GetParam(), { 203, 405 }, AVOCADO_DTYPE_FLOAT64);
			EXPECT_LT(data.getDifferenceLoss(), 1.0e-4);
		}
		TEST_P(TestLosses, float64_gradient)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			LossFunctionTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			EXPECT_LT(data.getDifferenceGradient(&alpha, &beta, true), 1.0e-4);
			EXPECT_LT(data.getDifferenceGradient(&alpha, &beta, false), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestLosses, TestLosses, ::testing::Values(AVOCADO_MEAN_SQUARE_LOSS, AVOCADO_CROSS_ENTROPY_LOSS, AVOCADO_KL_DIVERGENCE_LOSS));

	} /* namespace backend */
} /* namespace avocado */


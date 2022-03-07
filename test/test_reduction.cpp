/*
 * test_reduction.cpp
 *
 *  Created on: Jan 23, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestReduction: public testing::TestWithParam<avocado::backend::avReduceOp_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestReduction, float16)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			if (GetParam() >= AVOCADO_REDUCE_MUL and GetParam() <= AVOCADO_REDUCE_MAX)
				GTEST_SKIP(); // FIXME this mode doesn't work
			ReductionTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference1D(&alpha, &beta), 1.0e-2);
			EXPECT_LT(data.getDifferenceSingle(&alpha, &beta), 1.0e-2);
		}
		TEST_P(TestReduction, bfloat16)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			if ((GetParam() >= AVOCADO_REDUCE_MUL and GetParam() <= AVOCADO_REDUCE_MAX) or GetParam() == AVOCADO_REDUCE_MUL_NO_ZEROS)
				GTEST_SKIP(); // FIXME this mode doesn't work
			ReductionTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_BFLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference1D(&alpha, &beta), 1.0e-3);
			EXPECT_LT(data.getDifferenceSingle(&alpha, &beta), 1.0e-3);
		}
		TEST_P(TestReduction, float32)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			ReductionTester data(GetParam(), { 1, 1 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference1D(&alpha, &beta), 1.0e-3);
			EXPECT_LT(data.getDifferenceSingle(&alpha, &beta), 1.0e-3);
		}
		TEST_P(TestReduction, float64)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			ReductionTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			EXPECT_LT(data.getDifference1D(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceSingle(&alpha, &beta), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestReduction, TestReduction,
				::testing::Values(AVOCADO_REDUCE_ADD, AVOCADO_REDUCE_MUL, AVOCADO_REDUCE_MIN, AVOCADO_REDUCE_MAX, AVOCADO_REDUCE_AMAX, AVOCADO_REDUCE_AVG,
						AVOCADO_REDUCE_NORM1, AVOCADO_REDUCE_NORM2, AVOCADO_REDUCE_MUL_NO_ZEROS));

	} /* namespace backend */
} /* namespace avocado */


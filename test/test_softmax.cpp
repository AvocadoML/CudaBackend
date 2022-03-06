/*
 * test_softmax.cpp
 *
 *  Created on: Jan 24, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestSoftmax: public testing::TestWithParam<avocado::backend::avSoftmaxMode_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestSoftmax, float16)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			SoftmaxTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-2);
		}
		TEST_P(TestSoftmax, bfloat16)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			SoftmaxTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_BFLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-2);
		}
		TEST_P(TestSoftmax, float32)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			SoftmaxTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-3);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-3);
		}
		TEST_P(TestSoftmax, float64)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			SoftmaxTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestSoftmax, TestSoftmax, ::testing::Values(AVOCADO_SOFTMAX_MODE_INSTANCE, AVOCADO_SOFTMAX_MODE_CHANNEL));

	} /* namespace backend */
} /* namespace avocado */


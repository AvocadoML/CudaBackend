/*
 * test_activation.cpp
 *
 *  Created on: Jan 23, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestActivation: public testing::TestWithParam<avocado::backend::avActivationType_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestActivation, float16)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			ActivationTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-2);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-2);
		}
//		TEST_P(TestActivation, bfloat16)
//		{
//			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
//				GTEST_SKIP();
//			ActivationTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_BFLOAT16);
//			float alpha = 1.1f, beta = 0.1f;
//			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-2);
//			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-2);
//		}
		TEST_P(TestActivation, float32)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			ActivationTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-3);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-3);
		}
		TEST_P(TestActivation, float64)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			ActivationTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			EXPECT_LT(data.getDifferenceForward(&alpha, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceBackward(&alpha, &beta), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestActivation, TestActivation,
				::testing::Values(AVOCADO_ACTIVATION_LINEAR, AVOCADO_ACTIVATION_SIGMOID, AVOCADO_ACTIVATION_TANH, AVOCADO_ACTIVATION_RELU,
						AVOCADO_ACTIVATION_SELU, AVOCADO_ACTIVATION_ELU, AVOCADO_ACTIVATION_EXPONENTIAL, AVOCADO_ACTIVATION_SOFTPLUS,
						AVOCADO_ACTIVATION_SOFTSIGN));

	} /* namespace backend */
} /* namespace avocado */


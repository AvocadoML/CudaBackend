/*
 * test_unary_op.cpp
 *
 *  Created on: Jan 22, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestUnaryOp: public testing::TestWithParam<avocado::backend::avUnaryOp_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestUnaryOp, float16)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			UnaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-2);
		}
		TEST_P(TestUnaryOp, bfloat16)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			UnaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_BFLOAT16);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-3);
		}
		TEST_P(TestUnaryOp, float32)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			UnaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f, beta = 0.1f;
			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
		}
		TEST_P(TestUnaryOp, float64)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			UnaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1, beta = 0.1;
			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-3);
		}
		INSTANTIATE_TEST_SUITE_P(TestUnaryOp, TestUnaryOp,
				::testing::Values(AVOCADO_UNARY_OP_ABS, AVOCADO_UNARY_OP_CEIL, AVOCADO_UNARY_OP_COS, AVOCADO_UNARY_OP_EXP, AVOCADO_UNARY_OP_FLOOR,
						AVOCADO_UNARY_OP_LN, AVOCADO_UNARY_OP_NEG, AVOCADO_UNARY_OP_RCP, AVOCADO_UNARY_OP_RSQRT, AVOCADO_UNARY_OP_SIN, AVOCADO_UNARY_OP_SQUARE,
						AVOCADO_UNARY_OP_SQRT));//, AVOCADO_UNARY_OP_TAN)); // , AVOCADO_UNARY_OP_LOGICAL_NOT));

	} /* namespace backend */
} /* namespace avocado */


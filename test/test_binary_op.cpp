/*
 * test_binary_op.cpp
 *
 *  Created on: Jan 23, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestBinaryOp: public testing::TestWithParam<avocado::backend::avBinaryOp_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestBinaryOp, float16)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			BinaryOpTester data(GetParam(), { 2, 3 }, AVOCADO_DTYPE_FLOAT16);
			float alpha1 = 1.1f, alpha2 = 1.2f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceSame(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
			EXPECT_LT(data.getDifference1D(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
			EXPECT_LT(data.getDifferenceSingle(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
		}
		TEST_P(TestBinaryOp, bfloat16)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			BinaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_BFLOAT16);
			float alpha1 = 1.1f, alpha2 = 1.2f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceSame(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
			EXPECT_LT(data.getDifference1D(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
			EXPECT_LT(data.getDifferenceSingle(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT16));
		}
		TEST_P(TestBinaryOp, float32)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			BinaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float alpha1 = 1.1f, alpha2 = 1.2f, beta = 0.1f;
			EXPECT_LT(data.getDifferenceSame(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT32));
			EXPECT_LT(data.getDifference1D(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT32));
			EXPECT_LT(data.getDifferenceSingle(&alpha1, &alpha2, &beta), epsilonForTest(AVOCADO_DTYPE_FLOAT32));
		}
		TEST_P(TestBinaryOp, float64)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			BinaryOpTester data(GetParam(), { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
			double alpha1 = 1.1, alpha2 = 1.2, beta = 0.1;
			EXPECT_LT(data.getDifferenceSame(&alpha1, &alpha2, &beta), 1.0e-4);
			EXPECT_LT(data.getDifference1D(&alpha1, &alpha2, &beta), 1.0e-4);
			EXPECT_LT(data.getDifferenceSingle(&alpha1, &alpha2, &beta), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestBinaryOp, TestBinaryOp,
				::testing::Values(AVOCADO_BINARY_OP_ADD, AVOCADO_BINARY_OP_ADD_SQUARE, AVOCADO_BINARY_OP_SUB, AVOCADO_BINARY_OP_MUL, AVOCADO_BINARY_OP_DIV,
						AVOCADO_BINARY_OP_MOD, AVOCADO_BINARY_OP_POW, AVOCADO_BINARY_OP_MIN, AVOCADO_BINARY_OP_MAX));

//						,AVOCADO_BINARY_OP_COMPARE_EQ, AVOCADO_BINARY_OP_COMPARE_NEQ, AVOCADO_BINARY_OP_COMPARE_GT, AVOCADO_BINARY_OP_COMPARE_GE,
//						AVOCADO_BINARY_OP_COMPARE_LT, AVOCADO_BINARY_OP_COMPARE_LE, AVOCADO_BINARY_OP_LOGICAL_AND, AVOCADO_BINARY_OP_LOGICAL_OR,
//						AVOCADO_BINARY_OP_LOGICAL_XOR));

	} /* namespace backend */
} /* namespace avocado */


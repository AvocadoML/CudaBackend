/*
 * test_tensor_op.cpp
 *
 *  Created on: Jan 25, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestConcatSplitTranspose: public testing::TestWithParam<avocado::backend::avDataType_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST_P(TestConcatSplitTranspose, concat)
		{
			if (not supportsType( GetParam()))
				GTEST_SKIP();
			ConcatTester data( { 23, 45 }, GetParam());
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
		TEST_P(TestConcatSplitTranspose, split)
		{
			if (not supportsType( GetParam()))
				GTEST_SKIP();
			SplitTester data( { 23, 45 }, GetParam());
			EXPECT_LT(data.getDifference(), 1.0e-4);
		}
		TEST_P(TestConcatSplitTranspose, transpose)
		{
			if (not supportsType( GetParam()))
				GTEST_SKIP();
			TransposeTester data( { 23, 45, 67 }, GetParam());
			EXPECT_LT(data.getDifference( { 2, 0, 1 }), 1.0e-4);
		}
		INSTANTIATE_TEST_SUITE_P(TestConcatSplitTranspose, TestConcatSplitTranspose,
				::testing::Values(AVOCADO_DTYPE_INT8, AVOCADO_DTYPE_INT16, AVOCADO_DTYPE_INT32, AVOCADO_DTYPE_INT64, AVOCADO_DTYPE_FLOAT16,
						AVOCADO_DTYPE_BFLOAT16, AVOCADO_DTYPE_FLOAT32, AVOCADO_DTYPE_FLOAT64, AVOCADO_DTYPE_COMPLEX32, AVOCADO_DTYPE_COMPLEX64));

		TEST(TestTensorOp, float16_scale)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			ScaleTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT16);
			float alpha = 1.1f;
			EXPECT_LT(data.getDifference(&alpha), 1.0e-3);
		}
		TEST(TestTensorOp, bfloat16_scale)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			ScaleTester data( { 23, 45, 67 }, AVOCADO_DTYPE_BFLOAT16);
			float alpha = 1.1f;
			EXPECT_LT(data.getDifference(&alpha), 1.0e-3);
		}
		TEST(TestTensorOp, float32_scale)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			ScaleTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT32);
			float alpha = 1.1f;
			EXPECT_LT(data.getDifference(&alpha), 1.0e-4);
		}
		TEST(TestTensorOp, float64_scale)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			ScaleTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT64);
			double alpha = 1.1;
			EXPECT_LT(data.getDifference(&alpha), 1.0e-4);
		}

		TEST(TestTensorOp, float16_add_scalar)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			AddScalarTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT16);
			float scalar = 1.1f;
			EXPECT_LT(data.getDifference(&scalar), 1.0e-2);
		}
		TEST(TestTensorOp, bfloat16_add_scalar)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			AddScalarTester data( { 23, 45, 67 }, AVOCADO_DTYPE_BFLOAT16);
			float scalar = 1.1f;
			EXPECT_LT(data.getDifference(&scalar), 1.0e-2);
		}
		TEST(TestTensorOp, float32_add_scalar)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			AddScalarTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT32);
			float scalar = 1.1f;
			EXPECT_LT(data.getDifference(&scalar), 1.0e-4);
		}
		TEST(TestTensorOp, float64_add_scalar)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			AddScalarTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT64);
			double scalar = 1.1;
			EXPECT_LT(data.getDifference(&scalar), 1.0e-4);
		}

		TEST(TestTensorOp, float16_add_bias)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT16))
				GTEST_SKIP();
			AddBiasTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT16, AVOCADO_DTYPE_FLOAT16, AVOCADO_DTYPE_FLOAT32);
			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta1 = 0.1f;
			float beta2 = 0.2f;
			float beta3 = 0.3f;
			EXPECT_LT(data.getDifference(&alpha1, &alpha2, &beta1, &beta2, &beta3), 1.0e-3);
		}
		TEST(TestTensorOp, bfloat16_add_bias)
		{
			if (not supportsType(AVOCADO_DTYPE_BFLOAT16))
				GTEST_SKIP();
			AddBiasTester data( { 23, 45, 67 }, AVOCADO_DTYPE_BFLOAT16, AVOCADO_DTYPE_BFLOAT16, AVOCADO_DTYPE_FLOAT32);
			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta1 = 0.1f;
			float beta2 = 0.2f;
			float beta3 = 0.3f;
			EXPECT_LT(data.getDifference(&alpha1, &alpha2, &beta1, &beta2, &beta3), 1.0e-2);
		}
		TEST(TestTensorOp, float32_add_bias)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			AddBiasTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT32);
			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta1 = 0.1f;
			float beta2 = 0.2f;
			float beta3 = 0.3f;
			EXPECT_LT(data.getDifference(&alpha1, &alpha2, &beta1, &beta2, &beta3), 1.0e-4);
		}
		TEST(TestTensorOp, float64_add_bias)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			AddBiasTester data( { 23, 45, 67 }, AVOCADO_DTYPE_FLOAT64);
			double alpha1 = 1.1;
			double alpha2 = 1.2;
			double beta1 = 0.1;
			double beta2 = 0.2;
			double beta3 = 0.3;
			EXPECT_LT(data.getDifference(&alpha1, &alpha2, &beta1, &beta2, &beta3), 1.0e-4);
		}

	} /* namespace backend */
} /* namespace avocado */


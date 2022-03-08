/*
 * test_winograd.cpp
 *
 *  Created on: Feb 20, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace
{
	class TestWinogradTransforms: public testing::TestWithParam<avocado::backend::avDataType_t>
	{
	};
}

namespace avocado
{
	namespace backend
	{
		TEST(TestWinogradTransforms, transform_3x3_2x2_float16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_3x3_2x2_bfloat16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_BFLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_3x3_2x2_float32)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT32;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_3x3_2x2_float64)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT64;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			double alpha1 = 1.1f;
			double alpha2 = 1.2f;
			double beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}

		TEST(TestWinogradTransforms, transform_3x3_4x4_float16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 4);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_3x3_4x4_bfloat16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_BFLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 4);
			EXPECT_LT(data.getDifferenceWeight(), 0.2);
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), 0.05);
		}
		TEST(TestWinogradTransforms, transform_3x3_4x4_float32)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT32;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 4);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_3x3_4x4_float64)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT64;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 3, 3, 135 }, dtype, 4);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			double alpha1 = 1.1f;
			double alpha2 = 1.2f;
			double beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}

		TEST(TestWinogradTransforms, transform_5x5_2x2_float16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 5, 5, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_5x5_2x2_bfloat16)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_BFLOAT16;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 5, 5, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), 0.02);
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), 0.05);
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), 0.05);
		}
		TEST(TestWinogradTransforms, transform_5x5_2x2_float32)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT32;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 5, 5, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			float alpha1 = 1.1f;
			float alpha2 = 1.2f;
			float beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}
		TEST(TestWinogradTransforms, transform_5x5_2x2_float64)
		{
			const avDataType_t dtype = AVOCADO_DTYPE_FLOAT64;
			if (not supportsType(dtype))
				GTEST_SKIP();
			WinogradTest data( { 12, 13, 14, 135 }, { 21, 5, 5, 135 }, dtype, 2);
			EXPECT_LT(data.getDifferenceWeight(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceInput(), epsilonForTest(dtype));

			double alpha1 = 1.1f;
			double alpha2 = 1.2f;
			double beta = 0.1f;
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, true, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, true), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceOutput(&alpha1, &alpha2, &beta, false, false), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceGradient(), epsilonForTest(dtype));
			EXPECT_LT(data.getDifferenceUpdate(&alpha1, &beta), epsilonForTest(dtype));
		}

	} /* namespace backend */
} /* namespace avocado */


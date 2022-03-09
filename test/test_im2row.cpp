/*
 * test_im2row.cpp
 *
 *  Created on: Jan 27, 2022
 *      Author: Maciej Kozarzewski
 */

#include <testing/testing_helpers.hpp>

#include <gtest/gtest.h>

namespace avocado
{
	namespace backend
	{
		TEST(TestIm2Row2D, filter3x3)
		{
			Im2rowTest data( { 12, 23, 34, 45 }, { 11, 3, 3, 45 }, AVOCADO_DTYPE_FLOAT32);
			uint32_t padding[4] = { 0, 0, 0, 0 };
			data.set(AVOCADO_CONVOLUTION_MODE, { -1, -1, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, 1, padding);
			EXPECT_LE(data.getDifference(), 1.0e-6);
		}
		TEST(TestIm2Row2D, filter5x5)
		{
			Im2rowTest data( { 12, 23, 34, 45 }, { 11, 5, 5, 45 }, AVOCADO_DTYPE_FLOAT32);
			uint32_t padding[4] = { 0, 0, 0, 0 };
			data.set(AVOCADO_CONVOLUTION_MODE, { -2, -2, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, 1, padding);
			EXPECT_LE(data.getDifference(), 1.0e-6);
		}
//		TEST(TestIm2Row2D, padding_no_stride_no_dilation)
//		{
//			if (not supportsType(AVOCADO_DTYPE_INT32))
//				GTEST_SKIP();
//			Im2rowTest data( { 12, 23, 34, 45 }, { 11, 3, 3, 45 }, AVOCADO_DTYPE_INT32);
//			uint32_t padding[4] = { 0, 0, 0, 0 };
//			data.set(AVOCADO_CONVOLUTION_MODE, { -1, -2, 0 }, { 1, 1, 0 }, { 1, 1, 0 }, 1, padding);
//			EXPECT_LE(data.getDifference(), 1.0e-6);
//		}
//		TEST(TestIm2Row2D, no_padding_stride_no_dilation)
//		{
//			if (not supportsType(AVOCADO_DTYPE_INT32))
//				GTEST_SKIP();
//			Im2rowTest data( { 12, 23, 34, 45 }, { 11, 3, 3, 45 }, AVOCADO_DTYPE_INT32);
//			uint32_t padding[4] = { 0, 0, 0, 0 };
//			data.set(AVOCADO_CONVOLUTION_MODE, { 0, 0, 0 }, { 2, 2, 0 }, { 1, 1, 0 }, 1, padding);
//			EXPECT_LE(data.getDifference(), 1.0e-6);
//		}

	} /* namespace backend */
} /* namespace avocado */


/*
 * test_regularizers.cpp
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
		TEST(TestRegularizer, float32_L2)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT32))
				GTEST_SKIP();
			RegularizerTest data( { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
			float coefficient = 1.1f, offset = 0.1f;
			EXPECT_LT(data.getDifference(&coefficient, &offset), 1.0e-3);
		}
		TEST(TestRegularizer, float64_L2)
		{
			if (not supportsType(AVOCADO_DTYPE_FLOAT64))
				GTEST_SKIP();
			RegularizerTest data( { 203, 405 }, AVOCADO_DTYPE_FLOAT64);
			double coefficient = 1.1, offset = 0.1;
			EXPECT_LT(data.getDifference(&coefficient, &offset), 1.0e-4);
		}

	} /* namespace backend */
} /* namespace avocado */


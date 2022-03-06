/*
 * test_optimizers.cpp
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
//		TEST(TestOptimizer, float32_sgd)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT32))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.0, 0., 0.0, 0.0 }, { false, false, false, false });
//			float alpha = 1.1f, beta = 0.1f;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float32_sgd_momentum)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT32))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.01, 0.0, 0.0, 0.0 }, { true, false, false, false });
//			float alpha = 1.1f, beta = 0.1f;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float32_sgd_nesterov)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT32))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.01, 0.0, 0.0, 0.0 }, { true, true, false, false });
//			float alpha = 1.1f, beta = 0.1f;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float32_adam)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT32))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT32);
//			data.set(AVOCADO_OPTIMIZER_ADAM, 0.01, { 0.01, 0.001, 0.0, 0.0 }, { false, false, false, false });
//			float alpha = 1.1f, beta = 0.1f;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//
//		TEST(TestOptimizer, float64_sgd)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT64))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.0, 0., 0.0, 0.0 }, { false, false, false, false });
//			double alpha = 1.1, beta = 0.1;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float64_sgd_momentum)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT64))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.01, 0.0, 0.0, 0.0 }, { true, false, false, false });
//			double alpha = 1.1, beta = 0.1;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float64_sgd_nesterov)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT64))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
//			data.set(AVOCADO_OPTIMIZER_SGD, 0.01, { 0.01, 0.0, 0.0, 0.0 }, { true, true, false, false });
//			double alpha = 1.1, beta = 0.1;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}
//		TEST(TestOptimizer, float64_adam)
//		{
//			if (not supportsType(0, AVOCADO_DTYPE_FLOAT64))
//				GTEST_SKIP();
//			OptimizerTester data(0, { 23, 45 }, AVOCADO_DTYPE_FLOAT64);
//			data.set(AVOCADO_OPTIMIZER_ADAM, 0.01, { 0.01, 0.001, 0.0, 0.0 }, { false, false, false, false });
//			double alpha = 1.1, beta = 0.1;
//			EXPECT_LT(data.getDifference(&alpha, &beta), 1.0e-4);
//		}

	} /* namespace backend */
} /* namespace avocado */


/*
 * activations.cuh
 *
 *  Created on: Nov 23, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef ACTIVATIONS_CUH_
#define ACTIVATIONS_CUH_

#include "numbers/numbers.cuh"

template<typename T>
__host__   __device__ T scalar_zero() noexcept
{
	return static_cast<T>(0.0);
}
template<typename T>
__host__   __device__ T scalar_one() noexcept
{
	return static_cast<T>(1.0);
}
template<typename T>
__host__   __device__ T scalar_eps() noexcept
{
	return static_cast<T>(1.0e-16);
}

//template<typename dstType, typename srcType>
//struct Store
//{
//	__device__ dstType operator()(srcType x) const noexcept
//	{
//		return static_cast<dstType>(x);
//	}
//};
//template<>
//struct Store<int8_t, float>
//{
//	__device__ int8_t operator()(float x) const noexcept
//	{
//		return static_cast<int8_t>(max(-128.0f, min(127.0f, x)));
//	}
//};
//template<>
//struct Store<uint8_t, float>
//{
//	__device__ uint8_t operator()(float x) const noexcept
//	{
//		return static_cast<uint8_t>(max(0.0f, min(255.0f, x)));
//	}
//};
//template<>
//struct Store<int16_t, float>
//{
//	__device__ int16_t operator()(float x) const noexcept
//	{
//		return static_cast<int16_t>(max(-32768.0f, min(32767.0f, x)));
//	}
//};
//template<>
//struct Store<int32_t, float>
//{
//	__device__ int32_t operator()(float x) const noexcept
//	{
//		return static_cast<int32_t>(max(-2147483648.0f, min(2147483647.0f, x)));
//	}
//};

template<typename T>
__device__    __host__ constexpr T square(T x) noexcept
{
	return x * x;
}
template<typename T>
__device__ T safe_log(T x) noexcept
{
	return std::log(scalar_eps<T>() + x);
}
template<typename T>
__device__ bool ispow2(T x) noexcept
{
	return x > scalar_zero<T>() && !(x & (x - scalar_one<T>()));
}
template<typename T>
__device__ T sgn(T x) noexcept
{
	if (x > scalar_zero<T>())
		return scalar_one<T>();
	else
	{
		if (x < scalar_zero<T>())
			return -scalar_one<T>();
		else
			return scalar_zero<T>();
	}
//	return (zero<T>() < x) - (x < zero<T>());
}

/**
 * \brief Linear activation function.
 */
template<typename T>
struct ActivationLinear
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return input;
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient;
	}
//	__device__ T forward(T input) const
//	{
//		return input;
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient;
//	}
};

/**
 * \brief Sigmoidal activation function.
 */
template<typename T>
struct ActivationSigmoid
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return numbers::one<T>() / (numbers::one<T>() + numbers::exp(-input));
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient * (numbers::one<T>() - output) * output;
	}
//	__device__ T forward(T input) const
//	{
//		return scalar_one<T>() / (scalar_one<T>() + exp(-input));
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient * (scalar_one<T>() - output) * output;
//	}
};

/**
 * \brief Hyperbolic tangent activation function.
 */
template<typename T>
struct ActivationTanh
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return numbers::tanh(input);
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient * (numbers::one<T>() - output) * (numbers::one<T>() + output);
	}
//	__device__ T forward(T input) const
//	{
//		return tanh(input);
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient * (scalar_one<T>() - output) * (scalar_one<T>() + output);
//	}
};

/**
 * \brief Rectified linear unit activation function.
 */
template<typename T>
struct ActivationRelu
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return numbers::max(numbers::zero<T>(), input);
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return output > numbers::zero<T>() ? gradient : numbers::zero<T>();
	}
//	__device__ T forward(T input) const
//	{
//		return max(scalar_zero<T>(), input);
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return output > scalar_zero<T>() ? gradient : scalar_zero<T>();
//	}
};

/**
 * \brief Scaled linear unit activation function.
 */
template<typename T>
struct ActivationSelu
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		if (input >= numbers::zero<T>())
			return numbers::Number<T>(1.05070098f) * input;
		else
			return numbers::Number<T>(1.05070098f) * numbers::Number<T>(1.67326324f) * numbers::expm1(input);
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		if (output >= numbers::zero<T>())
			return numbers::Number<T>(1.05070098f) * gradient;
		else
			return numbers::Number<T>(1.05070098f) * gradient * numbers::Number<T>(1.67326324f) * (output + numbers::one<T>());
	}
//	__device__ T forward(T input) const
//	{
//		return static_cast<T>(1.05070098) * (input >= scalar_zero<T>() ? input : static_cast<T>(1.67326324) * expm1(input));
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return static_cast<T>(1.05070098) * gradient * (output >= scalar_zero<T>() ? scalar_one<T>() : static_cast<T>(1.67326324) * (output + scalar_one<T>()));
//	}
};

/**
 * \brief Exponential linear unit activation function.
 */
template<typename T>
struct ActivationElu
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		if (input >= numbers::zero<T>())
			return input;
		else
			return numbers::expm1(input);
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		if (output >= numbers::zero<T>())
			return gradient;
		else
			return gradient * (output + numbers::one<T>());
	}
//	__device__ T forward(T input) const
//	{
//		return input >= scalar_zero<T>() ? input : expm1(input);
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient * (output >= scalar_zero<T>() ? scalar_one<T>() : (output + scalar_one<T>()));
//	}
};

/**
 * \brief Exponential activation function.
 */
template<typename T>
struct ActivationExponential
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return numbers::exp(input);
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient * output;
	}
//	__device__ T forward(T input) const
//	{
//		return exp(input);
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient * output;
//	}
};

/**
 * \brief Softplus activation function.
 */
template<typename T>
struct ActivationSoftplus
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return numbers::log1p(numbers::exp(input));
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient * numbers::expm1(output) / numbers::exp(output);
	}
//	__device__ T forward(T input) const
//	{
//		return log1p(exp(input));
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient * expm1(output) / exp(output);
//	}
};

/**
 * \brief Softsign activation function.
 */
template<typename T>
struct ActivationSoftsign
{
	__device__ numbers::Number<T> forward(numbers::Number<T> input) const
	{
		return input / (numbers::abs(input) + numbers::one<T>());
	}
	__device__ numbers::Number<T> backward(numbers::Number<T> gradient, numbers::Number<T> output) const
	{
		return gradient / square(abs(output / (numbers::one<T>() - numbers::sgn(output) * output)) + numbers::one<T>());
	}
//	__device__ T forward(T input) const
//	{
//		return input / (fabs(input) + scalar_one<T>());
//	}
//	__device__ T backward(T gradient, T output) const
//	{
//		return gradient / square(fabs(output / (scalar_one<T>() - sgn(output) * output)) + scalar_one<T>());
//	}
};

namespace avocado
{
	namespace backend
	{
		template<typename T>
		__device__ numbers::Number<T> activation_forward(avActivationType_t activation, numbers::Number<T> input) noexcept
		{
			switch (activation)
			{
				default:
				case AVOCADO_ACTIVATION_LINEAR:
					return ActivationLinear<T>().forward(input);
				case AVOCADO_ACTIVATION_SIGMOID:
					return ActivationSigmoid<T>().forward(input);
				case AVOCADO_ACTIVATION_TANH:
					return ActivationTanh<T>().forward(input);
				case AVOCADO_ACTIVATION_RELU:
					return ActivationRelu<T>().forward(input);
				case AVOCADO_ACTIVATION_SELU:
					return ActivationSelu<T>().forward(input);
				case AVOCADO_ACTIVATION_ELU:
					return ActivationElu<T>().forward(input);
				case AVOCADO_ACTIVATION_EXPONENTIAL:
					return ActivationExponential<T>().forward(input);
				case AVOCADO_ACTIVATION_SOFTPLUS:
					return ActivationSoftplus<T>().forward(input);
				case AVOCADO_ACTIVATION_SOFTSIGN:
					return ActivationSoftsign<T>().forward(input);
			}
		}
		template<typename T>
		__device__ numbers::Number<T> activation_backward(avActivationType_t activation, numbers::Number<T> gradient, numbers::Number<T> output) noexcept
		{
			switch (activation)
			{
				default:
				case AVOCADO_ACTIVATION_LINEAR:
					return ActivationLinear<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_SIGMOID:
					return ActivationSigmoid<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_TANH:
					return ActivationTanh<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_RELU:
					return ActivationRelu<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_SELU:
					return ActivationSelu<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_ELU:
					return ActivationElu<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_EXPONENTIAL:
					return ActivationExponential<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_SOFTPLUS:
					return ActivationSoftplus<T>().backward(gradient, output);
				case AVOCADO_ACTIVATION_SOFTSIGN:
					return ActivationSoftsign<T>().backward(gradient, output);
			}
		}
	} /* namespace backend */
} /* namespace avocado */

#endif /* ACTIVATIONS_CUH_ */

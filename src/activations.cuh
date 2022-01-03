/*
 * activations.cuh
 *
 *  Created on: Nov 23, 2020
 *      Author: Maciej Kozarzewski
 */

#ifndef ACTIVATIONS_CUH_
#define ACTIVATIONS_CUH_

template<typename T>
__host__ __device__ T zero() noexcept
{
	return static_cast<T>(0);
}
template<typename T>
__host__ __device__ T one() noexcept
{
	return static_cast<T>(1);
}
template<typename T>
__host__ __device__ T eps() noexcept
{
	return static_cast<T>(1.0e-16);
}

template<typename dstType, typename srcType>
struct Store
{
	__device__ dstType operator()(srcType x) const noexcept
	{
		return static_cast<dstType>(x);
	}
};
template<>
struct Store<int8_t, float>
{
	__device__ int8_t operator()(float x) const noexcept
	{
		return static_cast<int8_t>(max(-128.0f, min(127.0f, x)));
	}
};
template<>
struct Store<uint8_t, float>
{
	__device__ uint8_t operator()(float x) const noexcept
	{
		return static_cast<uint8_t>(max(0.0f, min(255.0f, x)));
	}
};
template<>
struct Store<int16_t, float>
{
	__device__ int16_t operator()(float x) const noexcept
	{
		return static_cast<int16_t>(max(-32768.0f, min(32767.0f, x)));
	}
};
template<>
struct Store<int32_t, float>
{
	__device__ int32_t operator()(float x) const noexcept
	{
		return static_cast<int32_t>(max(-2147483648.0f, min(2147483647.0f, x)));
	}
};

template<typename T>
__device__ constexpr T square(T x) noexcept
{
	return x * x;
}
template<typename T>
__device__ T safe_log(T x) noexcept
{
	return std::log(eps<T>() + x);
}
template<typename T>
__device__ bool ispow2(T x) noexcept
{
	return x > zero<T>() && !(x & (x - one<T>()));
}
template<typename T>
__device__ T sgn(T x) noexcept
{
	return (zero<T>() < x) - (x < zero<T>());
}

/**
 * \brief Linear activation function.
 */
template<typename T>
struct ActivationLinear
{
	__device__ T forward(T input) const
	{
		return input;
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient;
	}
};

/**
 * \brief Sigmoidal activation function.
 */
template<typename T>
struct ActivationSigmoid
{
	__device__ T forward(T input) const
	{
		return one<T>() / (one<T>() + exp(-input));
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient * (one<T>() - output) * output;
	}
};

/**
 * \brief Hyperbolic tangent activation function.
 */
template<typename T>
struct ActivationTanh
{
	__device__ T forward(T input) const
	{
		return tanh(-input);
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient * (one<T>() - output) * (one<T>() + output);
	}
};

/**
 * \brief Rectified linear unit activation function.
 */
template<typename T>
struct ActivationRelu
{
	__device__ T forward(T input) const
	{
		return max(zero<T>(), input);
	}
	__device__ T backward(T gradient, T output) const
	{
		return output > zero<T>() ? gradient : zero<T>();
	}
};

/**
 * \brief Scaled linear unit activation function.
 */
template<typename T>
struct ActivationSelu
{
	__device__ T forward(T input) const
	{
		return static_cast<T>(1.05070098) * (input >= zero<T>() ? input : static_cast<T>(1.67326324) * expm1(input));
	}
	__device__ T backward(T gradient, T output) const
	{
		return static_cast<T>(1.05070098) * gradient * (output >= zero<T>() ? one<T>() : static_cast<T>(1.67326324) * (output + one<T>()));
	}
};

/**
 * \brief Exponential linear unit activation function.
 */
template<typename T>
struct ActivationElu
{
	__device__ T forward(T input) const
	{
		return input >= zero<T>() ? input : expm1(input);
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient * (output >= zero<T>() ? one<T>() : (output + one<T>()));
	}
};

/**
 * \brief Exponential activation function.
 */
template<typename T>
struct ActivationExponential
{
	__device__ T forward(T input) const
	{
		return exp(input);
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient * output;
	}
};

/**
 * \brief Softplus activation function.
 */
template<typename T>
struct ActivationSoftplus
{
	__device__ T forward(T input) const
	{
		return log1p(exp(input));
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient * expm1(output) / exp(output);
	}
};

/**
 * \brief Softsign activation function.
 */
template<typename T>
struct ActivationSoftsign
{
	__device__ T forward(T input) const
	{
		return input / (fabs(input) + one<T>());
	}
	__device__ T backward(T gradient, T output) const
	{
		return gradient / square(fabs(output / (one<T>() - sgn(output) * output)) + one<T>());
	}
};

namespace avocado
{
	namespace backend
	{
		template<typename T>
		__global__ void test_kernel(T *x)
		{
			x[0] = zero<T>();
		}

		template<typename T>
		__device__ T activation_forward(avActivationType_t activation, T input) noexcept
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
		__device__ T activation_backward(avActivationType_t activation, T gradient, T output) noexcept
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

/*
 * logical_ops.cuh
 *
 *  Created on: Dec 26, 2021
 *      Author: maciek
 */

#ifndef LOGICAL_OPS_CUH_
#define LOGICAL_OPS_CUH_

#include <cuda_runtime_api.h>

template<typename T, typename U>
__device__ T bit_cast(U x)
{
	return x;
}
template<>
__forceinline__ __device__ uint32_t bit_cast<uint32_t, float>(float x)
{
	return reinterpret_cast<uint32_t*>(&x)[0];
}
template<>
__forceinline__ __device__ float bit_cast<float, uint32_t>(uint32_t x)
{
	return reinterpret_cast<float*>(&x)[0];
}
template<>
__forceinline__ __device__ uint64_t bit_cast<uint64_t, double>(double x)
{
	return reinterpret_cast<uint64_t*>(&x)[0];
}
template<>
__forceinline__ __device__ double bit_cast<double, uint64_t>(uint64_t x)
{
	return reinterpret_cast<double*>(&x)[0];
}

#endif /* LOGICAL_OPS_CUH_ */

/*
 * regularizers.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	template<typename T>
	__global__ void kernel_regularizer_l2(T *gradient, const T *param, T scale, T offset, unsigned int elements)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] += scale * (param[i] - offset);
	}

	template<typename T>
	__device__ void block_reduce_linear(T *ptr) noexcept
	{
		assert(ispow2(blockDim.x));
		for (unsigned int i = blockDim.x / 2; i >= 1; i /= 2) // sum results stored in temporary array
		{
			if (threadIdx.x < i)
				ptr[threadIdx.x] += ptr[threadIdx.x + i];
			__syncthreads();
		}
	}
	template<typename T>
	__global__ void kernel_calculate_l2_loss_step1(T *dst, const T *param, T scale, T offset, unsigned int elements)
	{
		__shared__ T storage[1024];
		T acc = scalar_zero<T>();
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
			acc += square(param[i] - offset);
		storage[threadIdx.x] = acc;

		__syncthreads();
		block_reduce_linear(storage);
		if (threadIdx.x == 0)
			dst[blockIdx.x] = storage[0] * static_cast<T>(0.5) * scale;
	}
	template<typename T>
	__global__ void kernel_calculate_l2_loss_step2(T *dst)
	{
		block_reduce_linear(dst);
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t cudaRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t dwMem,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const void *scale, const void *offset, void *loss)
		{
			const unsigned int elements = getTensor(wDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(elements, blockDim.x));

			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			switch (getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_regularizer_l2<<<gridDim, blockDim, 0, stream>>>(getPointer<float>(dwMem), getPointer<float>(wMem), getScalarValue<float>(scale),
							getScalarValue<float>(offset), elements);
					if (loss != nullptr)
					{
						float *workspace = getContext(context).getWorkspace().data<float>();
						gridDim = dim3(gridSize<1024>(elements, 1024));
						kernel_calculate_l2_loss_step1<<<gridDim, 1024, 0, stream>>>(workspace, getPointer<float>(wMem), getScalarValue<float>(scale),
								getScalarValue<float>(offset), elements);
						if (gridDim.x > 1)
							kernel_calculate_l2_loss_step2<<<1, 1024, 0, stream>>>(workspace);
						cudaMemcpyAsync(loss, workspace, sizeof(float), cudaMemcpyDeviceToHost, getContext(context).getStream());
					}
					break;
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					kernel_regularizer_l2<<<gridDim, blockDim, 0, stream>>>(getPointer<double>(dwMem), getPointer<double>(wMem), getScalarValue<double>(scale),
							getScalarValue<double>(offset), elements);
					if (loss != nullptr)
					{
						double *workspace = getContext(context).getWorkspace().data<double>();
						gridDim = dim3(gridSize<1024>(elements, 1024));
						kernel_calculate_l2_loss_step1<<<gridDim, 1024, 0, stream>>>(workspace, getPointer<double>(wMem), getScalarValue<double>(scale),
								getScalarValue<double>(offset), elements);
						if (gridDim.x > 1)
							kernel_calculate_l2_loss_step2<<<1, 1024, 0, stream>>>(workspace);
						cudaMemcpyAsync(loss, workspace, sizeof(double), cudaMemcpyDeviceToHost, getContext(context).getStream());
					}
					break;
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}
	} /* namespace backend */
} /* namespace avocado */

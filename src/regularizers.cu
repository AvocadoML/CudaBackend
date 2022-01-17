/*
 * regularizers.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	template<typename T>
	__global__ void kernel_regularizer_l2(T *gradient, const T *param, T coefficient, T offset, unsigned int elements)
	{
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			gradient[i] += coefficient * (param[i] - offset);
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
	__global__ void kernel_calculate_l2_loss_step1(T *dst, const T *param, T coefficient, T offset, unsigned int elements)
	{
		__shared__ T storage[1024];
		T acc = zero<T>();
		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += blockDim.x * gridDim.x)
			acc += square(param[i] - offset);
		storage[threadIdx.x] = acc;

		__syncthreads();
		block_reduce_linear(storage);
		if (threadIdx.x == 0)
			dst[blockIdx.x] = storage[0] * static_cast<T>(0.5);
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
		avStatus_t cudaRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
				const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem, const void *coefficient, const void *offset, void *loss)
		{
			const unsigned int elements = cuda::getTensor(weightDesc).volume();
			dim3 gridDim(512);
			dim3 blockDim(256);
			cudaStream_t stream = cuda::getContext(context).getStream();

			switch (cuda::getTensor(weightDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					kernel_regularizer_l2<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(gradientMem), cuda::getPointer<float>(weightMem),
							cuda::getScalarValue<float>(coefficient), cuda::getScalarValue<float>(offset), elements);
					if (loss != nullptr)
					{
						float *workspace = cuda::getContext(context).getWorkspace().data<float>();
						dim3 gridDim(gridSize<1024>(elements, 1024));
						kernel_calculate_l2_loss_step1<<<gridDim, 1024, 0, stream>>>(workspace, cuda::getPointer<float>(weightMem),
								cuda::getScalarValue<float>(coefficient), cuda::getScalarValue<float>(offset), elements);
						if (gridDim.x > 1)
							kernel_calculate_l2_loss_step2<<<1, 1024, 0, stream>>>(workspace);
						cudaMemcpyAsync(loss, workspace, sizeof(float), cudaMemcpyDeviceToHost, cuda::getContext(context).getStream());
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

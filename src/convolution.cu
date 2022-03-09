/*
 * convolution.cu
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
#include <iostream>

namespace
{
	using namespace avocado::backend;
}

namespace avocado
{
	namespace backend
	{

		avStatus_t cudaGetConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, av_int64 *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation, avMemoryDescriptor_t workspaceMem)
		{
			cuda::getContext(context).setDevice();
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t workspaceMem)
		{
			cuda::getContext(context).setDevice();
			switch (cuda::getTensor(xDesc).dtype())
			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
//							reinterpret_cast<const half*>(weight->data), filters_in, invert);
//					break;
				case AVOCADO_DTYPE_FLOAT32:
					break;
				case AVOCADO_DTYPE_FLOAT64:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
//							reinterpret_cast<const double*>(weight->data), filters_in, invert);
					break;
			}

//			uint4 shape { getTensor(xDesc).dimension(0), getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//			int2 padding { -1, -1 };
//			dim3 blockDim(128);
//			dim3 gridDim(shape.x, (shape.y + 3) / 4, (shape.z + 3) / 4);
//			cudaStream_t stream = getContext(context).getStream();
//
//			switch (getTensor(xDesc).dtype())
//			{
////				case AVOCADO_DTYPE_FLOAT16:
////					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
////							reinterpret_cast<const half*>(weight->data), filters_in, invert);
////					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_winograd_input_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(workspaceMem), getPointer<float>(xMem),
//							shape, padding, 0.0f);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
////					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
////							reinterpret_cast<const double*>(weight->data), filters_in, invert);
//					break;
//			}
//			cudaStreamSynchronize(stream);

//			int3 shape { getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//
//			int tiles_h = (getTensor(xDesc).dimension(1) + 3) / 4;
//			int tiles_w = (getTensor(xDesc).dimension(2) + 3) / 4;
//			dim3 gridSize(getTensor(xDesc).firstDim(), tiles_h, tiles_w);
//			cudaStream_t stream = getContext(context).getStream();
//
//			dim3 blockSize(32, 6);
//			kernel_conv3x3_4x4_input_transform<64> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(workspaceMem), getPointer<float>(xMem), shape);
//			cudaStreamSynchronize(stream);
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
				const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
			cuda::getContext(context).setDevice();
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

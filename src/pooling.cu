/*
 * pooling.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{

}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t cudaPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem)
		{
			getContext(context).setDevice();
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem)
		{
			getContext(context).setDevice();
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

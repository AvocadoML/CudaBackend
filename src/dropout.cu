/*
 * dropout.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		DLL_PUBLIC avStatus_t cudaDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t states)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

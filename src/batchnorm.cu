/*
 * batchnorm.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

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
		avStatus_t cudaAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t biasMem,
				avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const void *beta,
				const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc, avMemoryDescriptor_t dyMem,
				const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem, const avMemoryDescriptor_t meanMem,
				const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaBatchNormUpdate(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem,
				const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t scaleBiasDesc,
				avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem, const avMemoryDescriptor_t meanMem,
				const avMemoryDescriptor_t varianceMem, double epsilon)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

/*
 * metrics.cu
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
		avStatus_t cudaMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

/*
 * optimizers.cu
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
		avStatus_t cudaOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config, const avTensorDescriptor_t wDesc,
				avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem, avMemoryDescriptor_t workspace)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

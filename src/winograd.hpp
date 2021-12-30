/*
 * winograd.hpp
 *
 *  Created on: Dec 29, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef WINOGRAD_KERNELS_HPP_
#define WINOGRAD_KERNELS_HPP_

namespace avocado
{
	namespace backend
	{
		avSize_t winogradGetWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc);

		avStatus_t winogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem);

		avStatus_t winogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem);

		avStatus_t winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation);

		avStatus_t winogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem);

		avStatus_t cudaWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem);
	} /* namespace backend */
} /* namespace avocado */

#endif /* WINOGRAD_KERNELS_HPP_ */

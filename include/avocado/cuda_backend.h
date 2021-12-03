/*
 * cuda_backend.h
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef AVOCADO_CUDA_BACKEND_H_
#define AVOCADO_CUDA_BACKEND_H_

#include <avocado/backend/backend_defs.h>

namespace avocado
{
	namespace backend
	{
#ifdef __cplusplus
		extern "C"
		{
#endif
		/**
		 * All scaling factors are optional (unless specified otherwise) and will then behave as following:\n
		 * for alpha-like types the default value is 1.
		 * for beta-like types the default value is 0.
		 * The type for alpha and beta parameters must match the types of tensors with the exceptions for:
		 *  - all integer types - alpha and beta type must be float32. Unless specified otherwise, the integer value will be casted to float32,
		 *  scaling will be performed on float32, and then the value will be casted back to appropriate integer type.
		 *  - float16, bfloat16 - alpha and beta must be float32
		 */

		/**
		 * In few methods context can be null, the operation will be then performed in a synchronous way, potentially blocking other operations.
		 * But in most methods context is mandatory and must not be null.
		 * Context specifies the device on which the operation is performed.
		 */

		/* --------------------------------------------------------------------
		 * Implemented in 'cuda_features.cpp'
		 * --------------------------------------------------------------------
		 */
		DLL_PUBLIC struct CudaFeatures
		{
				char name[256];
				long long global_memory; /**< in bytes */
				long long shared_memory; /**< in bytes */
				int sm_count;
				int major;
				int minor;
				bool supports_dp4a;
				bool supports_fp16;
				bool supports_bfloat16;
				bool supports_fp64;
				bool has_tensor_cores;
		};
		DLL_PUBLIC avStatus_t cudaGetFeatures(CudaFeatures *result, avDeviceIndex_t deviceIndex);
		DLL_PUBLIC const char* cudaDecodeStatus(avStatus_t status);
		DLL_PUBLIC avStatus_t cudaGetNumberOfDevices(int *result);
		/**
		 * \brief Checks if peer-to-peer copy is possible between the two specified GPUs.
		 * If both device indices are the same, true is returned.
		 */
		DLL_PUBLIC avStatus_t cudaIsCopyPossible(bool *result, avDeviceIndex_t from, avDeviceIndex_t to);

		/* --------------------------------------------------------------------
		 * Implemented in 'context.cpp' and 'context.hpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief Creates new context associated with the specified device.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully created.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed pointer is null.
		 */
		DLL_PUBLIC avStatus_t cudaCreateContext(avContext_t *context, avDeviceIndex_t deviceIndex);
		/**
		 * \brief Destroys context. If null pointer is passed, the function does nothing.
		 *
		 * \param[in] context Pointer to a context to be destroyed.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully destroyed.
		 */
		DLL_PUBLIC avStatus_t cudaDestroyContext(avContext_t context);
		/**
		 * \brief Blocks until all operations in a given context are finished.
		 *
		 * \param[in] context Pointer to a context to synchronize with.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The synchronization was successfully performed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is a null pointer.
		 */
		DLL_PUBLIC avStatus_t cudaSynchronizeWithContext(avContext_t context);
		/**
		 * \brief Blocks until all operations in a given context are finished.
		 *
		 * \param[in] context Pointer to a context.
		 * \param[out] result
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The synchronization was successfully performed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is a null pointer.
		 */
		DLL_PUBLIC avStatus_t cudaIsContextReady(avContext_t context, bool *result);
		/**
		 * \brief Changes the workspace size of a given context.
		 *
		 * \param[in] context Context owning the workspace to change.
		 * \param[in] newSize New size of the workspace memory (in bytes).
		 * \param[in] forceShrink If the actual workspace size is lower than newSize and forceShrink is set to true, the workspace will be shrinked.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The workspace resize was successfully performed.
		 * \retval AVOCADO_STATUS_BAD_ALLOC The allocation of the new workspace failed. The previous workspace remains unchanged.
		 */
		DLL_PUBLIC avStatus_t cudaResizeWorkspace(avContext_t context, avSize_t newSize, bool forceShrink);
		/**
		 * \brief Returns pointer to the workspace managed by given context. Also returns size of this workspace.
		 * This method is unsafe to use. If for any reason the workspace changes, the pointer obtained via this method will be silently invalidated.
		 *
		 * \param[in] context Context owning the workspace.
		 * \param[out] ptr Pointer to a workspace pointer. Can be null, will be ignored then.
		 * \param[out] size Pointer to the 64bit integer where the size of workspace will be written. Can be null, will be ignored then.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The workspace size was successfully read.
		 */
		DLL_PUBLIC avStatus_t cudaGetWorkspace(avContext_t context, void **ptr, avSize_t *size);

		/* --------------------------------------------------------------------
		 * Implemented in 'memory.cpp' and 'memory.hpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief Allocates memory.
		 *
		 * \param[out] ptr Pointer to newly allocated block of memory.
		 * \param[in] count Number of bytes to allocate.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully allocated.
		 * \retval AVOCADO_STATUS_BAD_ALLOC The allocation failed.
		 */
		DLL_PUBLIC avStatus_t cudaAllocateMemory(void **ptr, avDeviceIndex_t deviceIndex, avSize_t count);
		/**
		 * \brief Frees memory.
		 *
		 * \param[out] ptr Pointer to block of memory to be deleted. Can be null, the function does nothing then.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully deleted.
		 */
		DLL_PUBLIC avStatus_t cudaFreeMemory(void *ptr, avDeviceIndex_t deviceIndex);
		/**
		 * \brief Sets memory with given pattern of bytes.
		 *
		 * \param[in] context Context in which the operation is performed. Can be null, the operation will be performed in the default context.
		 * \param[out] dst Destination pointer.
		 * \param[in] dstSize Number of bytes in the destination block.
		 * \param[in] pattern Pointer to pattern to be set. Can be null, the destination memory is zeroed then and the value patternSize argument is ignored.
		 * \param[in] patternSize Number of bytes of the pattern.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully set.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 the dst pointer is null.\n
		 the dstSize is not a multiple of patternSize.
		 */
		DLL_PUBLIC avStatus_t cudaSetMemory(avContext_t context, void *dst, avSize_t dstSize, const void *pattern, avSize_t patternSize);
		/**
		 * \brief Copies block of memory.
		 *
		 * \param[in] context Context in which the operation is performed. Can be null, the operation will be performed in the default context.
		 * \param[out] dst Destination pointer.
		 * \param[in] src Source pointer.
		 * \param[in] count Number of bytes to copy.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully copied.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 the dst pointer is null.\n
		 the src pointer is null.
		 */
		DLL_PUBLIC avStatus_t cudaCopyMemory(avContext_t context, void *dst, avDeviceIndex_t dstIndex, const void *src, avDeviceIndex_t srcIndex,
				avSize_t count);

		/*
		 *
		 * --------------------------------------------------------------------
		 * Implemented in 'conversions.cu' and 'basic_math.cu'.
		 * --------------------------------------------------------------------
		 *
		 *
		 */

		/**
		 * \brief This routine is used to convert between data types.
		 *
		 */
		DLL_PUBLIC avStatus_t cudaChangeType(avContext_t context, void *dst, avDataType_t dstType, const void *src, avDataType_t srcType,
				avSize_t elements);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaConcatTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaSplitTensors(avContext_t context, avTensor_t dst, const avTensor_t src, avSize_t lastDimOffsetInBytes);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaTranspose(avContext_t context, avTensor_t dst, const avTensor_t src, const int order[]);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaScaleTensor(avContext_t context, avTensor_t dst, const avScalar_t src);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaAddScalarToTensor(avContext_t context, avTensor_t dst, const avScalar_t src);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaOpTensor(avContext_t context, avOpTensorOp_t operation, const avScalar_t alpha1, const avTensor_t input1,
				const avScalar_t alpha2, const avTensor_t input2, const avScalar_t beta, avTensor_t output);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaOpSingleTensor(avContext_t context, avOpSingleTensorOp_t operation, const avScalar_t alpha, const avTensor_t input,
				const avScalar_t beta, avTensor_t output);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaReduceTensor(avContext_t context, avReduceTensorOp_t operation, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaAddTensors(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, avActivationType_t activation);

		/* --------------------------------------------------------------------
		 * Implemented in 'gemms.cpp'.
		 * --------------------------------------------------------------------
		 */

		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t cudaGemm(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta);
		/**
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t cudaGemmBatched(avContext_t context, avGemmOperation_t opA, avGemmOperation_t opB, avTensor_t C, const avTensor_t A,
				const avTensor_t B, const avScalar_t alpha, const avScalar_t beta);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'activations.cpp' and 'activations.hpp'
		 * --------------------------------------------------------------------
		 */

		/** \brief This routine applies a specified neuron activation function element-wise over each input value.
		 * In-place operation is allowed for this routine - input and output tensor pointers may be equal.
		 *
		 * \param[in]	context					Pointer to a previously created context. For more information, see avContext_t.
		 * \param[in]	activation				Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in]	alpha1, alpha2, beta	Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 * output = alpha1[0] * activation(alpha2[0] * input) + beta[0] * priorOutputValue
		 * \param[in]	input					Descriptor of input tensor.
		 * \param[out]	output					Descriptor of output tensor.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaActivationForward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha1,
				const avScalar_t alpha2, const avScalar_t beta, const avTensor_t input, avTensor_t output);

		/** \brief This routine calculates gradient of a specified neuron activation function.
		 * In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.
		 *
		 * \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 * \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 * 		dstValue = alpha * result + beta * priorDstValue
		 * \param[in] gradientNext Descriptor of gradient tensor after the layer.
		 * \param[in] output Descriptor of output tensor after the layer.
		 * \param[out] gradientPrev Descriptor of gradient tensor before the layer.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaActivationBackward(avContext_t context, const avActivationType_t activation, const void *alpha, const void *beta,
				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output);
		/**
		 * \brief This routine applies softmax function.
		 In-place operation is allowed for this routine - input and output tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] mode Mode indicating over which dimension the function is computed. For more information, see avSoftmaxMode_t.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] input Descriptor of input tensor.
		 \param[out] output Descriptor of output tensor.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaSoftmaxForward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		/**
		 * \brief This routine calculates gradient of the softmax function.
		 In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.

		 \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 \param[in] mode Mode indicating over which dimension the function is computed. For more information, see avSoftmaxMode_t.
		 \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 dstValue = alpha * result + beta * priorDstValue
		 \param[in] gradientNext Descriptor of gradient tensor after the layer.
		 \param[in] output Descriptor of output tensor after the layer.
		 \param[out] gradientPrev Descriptor of gradient tensor before the layer.

		 \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 The parameter mode has an invalid enumerant value.\n
		 The dimensions of the input tensor and output tensor differ.\n
		 The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaSoftmaxBackward(avContext_t context, avSoftmaxMode_t mode, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradientPrev, const avTensor_t gradientNext, const avTensor_t output);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'batch_norm.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaAffineForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t weight, const avTensor_t bias, const avActivationType_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormInference(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, const avTensor_t estimatedMean, const avTensor_t estimatedVariance,
				double epsilon, const avActivationType_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormForward(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				avTensor_t output, const avTensor_t scale, const avTensor_t bias, avTensor_t savedMean, avTensor_t savedVariance, double epsilon,
				const avActivationType_t activation);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormBackward(avContext_t context, const avActivationType_t activation, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t input, const avTensor_t output, avTensor_t gradientPrev, avTensor_t gradientNext,
				const avTensor_t scale, const avTensor_t savedMean, const avTensor_t savedVariance, double epsilon);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormUpdate(avContext_t context, const avScalar_t alpha, const avScalar_t beta, const avTensor_t input,
				const avTensor_t gradientNext, avTensor_t scaleUpdate, avTensor_t biasUpdate, const avTensor_t savedMean,
				const avTensor_t savedVariance, double epsilon);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'dropout.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaDropoutForward(avContext_t context, const avDropout_t config, const avTensor_t input, avTensor_t output,
				avTensor_t states);
		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaDropoutBackward(avContext_t context, const avDropout_t config, avTensor_t gradientPrev,
				const avTensor_t gradientNext, const avTensor_t states);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'pooling.cpp'
		 * --------------------------------------------------------------------
		 */

		DLL_PUBLIC avStatus_t cudaPoolingForward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output);
		DLL_PUBLIC avStatus_t cudaPoolingBackward(avContext_t context, const avPooling_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t gradientPrev, const avTensor_t gradientNext);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'winograd_math.cpp', 'im2col.cpp' and conv.cpp
		 * --------------------------------------------------------------------
		 */

		DLL_PUBLIC avStatus_t cudaWinogradWeightTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t weights,
				avTensor_t matrices);
		DLL_PUBLIC avStatus_t cudaWinogradInputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avTensor_t input,
				avTensor_t matrices);
		DLL_PUBLIC avStatus_t cudaWinogradOutputTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t output, const avTensor_t bias, const avActivationType_t activation);
		DLL_PUBLIC avStatus_t cudaWinogradGradientTransform(avContext_t context, const avConvolution_t config, int tileSize,
				const avTensor_t gradient, avTensor_t matrices);
		DLL_PUBLIC avStatus_t cudaWinogradUpdateTransform(avContext_t context, const avConvolution_t config, int tileSize, const avScalar_t alpha,
				const avScalar_t beta, const avTensor_t matrices, avTensor_t update);

		DLL_PUBLIC avStatus_t cudaIm2Col(avContext_t context, const avConvolution_t config, const avTensor_t input, avTensor_t output);

		/**
		 * output = activation(alpha1 * convolve(input, weights) + alpha2 * add + bias)
		 */
		DLL_PUBLIC avStatus_t cudaConvolutionBiasActivationForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha1,
				const avScalar_t beta, const avTensor_t input, avTensor_t output, const avTensor_t weights, const avTensor_t bias,
				const avActivationType_t activation, const avScalar_t alpha2, const avTensor_t add);
		/**
		 * \brief Simplified version of the above method.
		 * output = alpha * convolve(input, weights) + beta * output
		 */
		DLL_PUBLIC avStatus_t cudaConvolutionForward(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, avTensor_t output, const avTensor_t weights);
		DLL_PUBLIC avStatus_t cudaConvolutionBackward(avContext_t context, const avConvolution_t config, const avScalar_t alpha,
				const avScalar_t beta, avTensor_t gradientPrev, avTensor_t gradientNext, const avTensor_t output, const avTensor_t weights,
				const avActivationType_t activation);
		DLL_PUBLIC avStatus_t cudaConvolutionUpdate(avContext_t context, const avConvolution_t config, const avScalar_t alpha, const avScalar_t beta,
				const avTensor_t input, const avTensor_t gradientNext, avTensor_t weightUpdate, avTensor_t biasUpdate);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'metrics.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief Computes chosen metric function, averaged over entire batch.
		 */
		DLL_PUBLIC avStatus_t cudaMetricFunction(avContext_t context, avMetricType_t metricType, avScalar_t result, const avTensor_t output,
				const avTensor_t target);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'losses.cpp'
		 * --------------------------------------------------------------------
		 */
		/**
		 * \brief Computes one of the built-in loss functions.

		 * \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 * \param[in] lossType Enumeration specifying which loss function to calculate.
		 * \param[out] result Pointer to the resulting loss value.
		 * \param[in] output Output from a learning model.
		 * \param[in] target Target values for the learning model.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_UNSUPPORTED_DATATYPE The function does not support the provided data types.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * Any of the parameters is null.\n
		 * The parameter lossType has an invalid enumerant value.\n
		 * The dimensions of the output tensor and target tensor differ.\n
		 * The data type of the output tensor and target tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaLossFunction(avContext_t context, avLossType_t lossType, avScalar_t result, const avTensor_t output,
				const avTensor_t target);
		/**
		 * \brief Computes gradient of one of the built-in loss functions.

		 * \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 * \param[in] lossType Enumeration specifying which loss function to calculate.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 * 		dstValue = alpha * result + beta * priorDstValue
		 * \param[out] gradient Pointer to the gradient tensor.
		 * \param[in] output Output from a learning model.
		 * \param[in] target Target values for the learning model.
		 * \param[in] isFused Some loss functions can be fused with preceding activation layer in the backpropagation phase for better numerical stability.
		 * 		This flag toggles this fused mode.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_UNSUPPORTED_DATATYPE The function does not support the provided data types.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * Any of the parameters is null.\n
		 * The parameter lossType has an invalid enumerant value.\n
		 * The dimensions of the output tensor and target tensor differ.\n
		 * The data type of the output tensor and target tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaLossGradient(avContext_t context, avLossType_t lossType, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t gradient, const avTensor_t output, const avTensor_t target, bool isFused);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'optimizers.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 *
		 */
		DLL_PUBLIC avStatus_t cudaOptimizerLearn(avContext_t context, const avOptimizer_t optimizer, const avScalar_t alpha, const avScalar_t beta,
				avTensor_t weight, const avTensor_t update, avTensor_t workspace1, avTensor_t workspace2);

		/*
		 * --------------------------------------------------------------------
		 * implemented in 'regularizers.cpp'
		 * --------------------------------------------------------------------
		 */

		/**
		 * \brief Applies L2 regularization to the weight tensor.
		 * The exact formulas are like below.
		 * loss = coefficient * square(weight - offset)
		 * gradient += coefficient * (weight - offset)
		 *
		 * \param[in] context Pointer to a previously created context. For more information, see avContext_t.
		 * \param[out] gradient Pointer to gradient tensor which will be altered by the regularization gradient.
		 * \param[in] weight Pointer to the weight tensor.
		 * \param[in] coefficient Regularization strength.
		 * \param[in] offset Offset to the weights. This parameter is optional and can be null, the value of 0 will be used then.
		 * \param[out] loss If this parameter is not null, the total L2 loss will be computed in addition to the gradients.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_UNSUPPORTED_DATATYPE The function does not support the provided data types.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * Any of the mandatory parameters is null.\n
		 * The dimensions of the gradient tensor and weight tensor differ.\n
		 * The data type of the gradient tensor and weight tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaRegularizerL2(avContext_t context, avTensor_t gradient, const avTensor_t weight, const avScalar_t coefficient,
				const avScalar_t offset, avScalar_t loss);

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_CUDA_BACKEND_H_ */

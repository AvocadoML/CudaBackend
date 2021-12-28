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
		 * A few words about argument types. \n
		 * Descriptor types are passed by value, const keyword is used as a hint that object associated with the descriptor will not change within the function.
		 * All pointer and array types are assumed to be pointing to host memory.
		 *
		 * A few words about argument names. \n
		 *
		 * For functions for neural network layers there are 8 types or names: \n
		 * Argument name | Meaning
		 * ------------- | -------------
		 * x, dx         | input tensor, gradient at the input
		 * y, dy         | output tensor, gradient at the output
		 * w, dw         | weight tensor, gradient of weights
		 * b, db         | bias tensor, gradient of bias
		 * z             | another input to be somehow used by the function
		 *
		 * For other kinds of functions, letters 'a' and 'b' usually indicate inputs to the function, while letter 'c' indicates the output.
		 * Typically they followed by 'Desc' for tensor descriptors, 'Mem' for memory descriptors.
		 * Sometimes there may be more than one letter in the tensor descriptor name, like 'xyDesc'. It means that both 'x' and 'y' arguments have the same descriptor.
		 *
		 * In few functions output is named 'dst' while input is 'src'.
		 *
		 * Unless specified otherwise, all scaling factors are optional (can be null pointers) and will then behave as following:\n
		 * for alpha-like types the default value is 1.
		 * for beta-like types the default value is 0.
		 * The type for alpha and beta parameters must match the types of tensors with the exceptions for:
		 *  - all integer types - alpha and beta type must be float32. Unless specified otherwise, the integer tensor elements will be casted to float32,
		 *  scaling will be performed in float32, and then the element will be casted back to appropriate integer type.
		 *  - float16, bfloat16 - alpha and beta must be float32
		 *
		 * Context specifies the device on which the operation is performed.
		 */

		/**
		 * \brief Queries CPU device properties.
		 *
		 * \param[in] index Index of the device to query.
		 * \param[in] property Name Name of device property to read.
		 * \param[out] result Pointer to at least 256 bytes of memory.
		 */
		DLL_PUBLIC avStatus_t cudaGetDeviceProperty(avDeviceIndex_t index, avDeviceProperty_t propertyName, void *result);

		/**
		 * \brief Checks if a direct copy between devices is possible.
		 *
		 * \param[in] from Index of source device.
		 * \param[in] to Index of destination device.
		 * \param[out] result Pointer to the returned value.
		 */
		DLL_PUBLIC avStatus_t cudaIsCopyPossible(avDeviceIndex_t from, avDeviceIndex_t to, bool *result);

		/**
		 * \brief Creates new context.
		 *
		 * \param[out] result
		 * \param[in] deviceIndex
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully created.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed pointer is null.
		 */
		DLL_PUBLIC avStatus_t cudaCreateContextDescriptor(avContextDescriptor_t *result, avDeviceIndex_t deviceIndex);

		/**
		 * \brief Destroys context. If null pointer is passed, the function does nothing.
		 *
		 * \param[in] context Context descriptor to be destroyed.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The context was successfully destroyed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is invalid or is a descriptor of the default context.
		 * \retval AVOCADO_STATUS_FREE_FAILED Deallocation failed.
		 */
		DLL_PUBLIC avStatus_t cudaDestroyContextDescriptor(avContextDescriptor_t desc);

		/**
		 * \brief Returns default context for given device.
		 * This method never fails, so it returns the result directly.
		 *
		 * \param[in] deviceIndex
		 */
		DLL_PUBLIC avContextDescriptor_t cudaGetDefaultContext(avDeviceIndex_t deviceIndex);

		/**
		 * \brief Blocks until all operations in a given context are finished.
		 *
		 * \param[in] context Context descriptor to synchronize with.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The synchronization was successfully performed.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed context is invalid.
		 */
		DLL_PUBLIC avStatus_t cudaSynchronizeWithContext(avContextDescriptor_t context);

		/**
		 * \brief Checks if all operations in a given context are finished.
		 *
		 * \param[in] context Context descriptor to query for readiness.
		 * \param[out] result
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The readiness was successfully checked.
		 * \retval AVOCADO_STATUS_BAD_PARAM The result pointer is null.
		 */
		DLL_PUBLIC avStatus_t cudaIsContextReady(avContextDescriptor_t context, bool *result);

		/**
		 * \brief Allocates new memory block and creates its descriptor.
		 *
		 * \param[out] result Pointer to new memory descriptor.
		 * \param[in] sizeInBytes Number of bytes to allocate.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully allocated.
		 * \retval AVOCADO_STATUS_BAD_PARAM The passed pointer is null.
		 * \retval AVOCADO_STATUS_BAD_ALLOC The allocation failed.
		 */
		DLL_PUBLIC avStatus_t cudaCreateMemoryDescriptor(avMemoryDescriptor_t *result, avDeviceIndex_t deviceIndex, avSize_t sizeInBytes);

		/**
		 * \brief Creates non-owning view of another memory block.
		 *
		 * \param[out] result Pointer to the new memory descriptor
		 * \param[in] desc Original memory block to create view.
		 * \param[in] sizeInBytes Size of the view, in bytes (obviously).
		 * \param[in] offsetInBytes Offset relative to the beginning of the original memory block, in bytes.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory view was successfully created.
		 * \retval AVOCADO_STATUS_BAD_PARAM The descriptor is invalid or not owning or offset is negative.
		 */
		DLL_PUBLIC avStatus_t cudaCreateMemoryView(avMemoryDescriptor_t *result, const avMemoryDescriptor_t desc, avSize_t sizeInBytes,
				avSize_t offsetInBytes);

		/**
		 * \brief Frees memory and destroys the memory descriptor.
		 *
		 * \param[out] desc Memory descriptor to be deleted.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully deleted.
		 * \retval AVOCADO_STATUS_FREE_FAILED Deallocation failed.
		 */
		DLL_PUBLIC avStatus_t cudaDestroyMemoryDescriptor(avMemoryDescriptor_t desc);

		/**
		 * \brief Sets memory with given pattern of bytes.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst Destination memory block.
		 * \param[in] dstSize Number of bytes in the destination block.
		 * \param[in] pattern Pointer to pattern to be set. Can be null, the destination memory is zeroed then and the value patternSize argument is ignored.
		 * \param[in] patternSize Number of bytes of the pattern.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully set.
		 * \retval AVOCADO_STATUS_BAD_PARAM The dstSize is not a multiple of patternSize.
		 */
		DLL_PUBLIC avStatus_t cudaSetMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, avSize_t dstSize, const void *pattern,
				avSize_t patternSize);

		/**
		 * \brief Copies block of memory.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst Destination pointer.
		 * \param[in] src Source pointer.
		 * \param[in] count Number of bytes to copy.
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The memory was successfully copied.
		 * \retval AVOCADO_STATUS_BAD_PARAM Either dst descriptor or src descriptor is invalid.
		 */
		DLL_PUBLIC avStatus_t cudaCopyMemory(avContextDescriptor_t context, avMemoryDescriptor_t dst, const avMemoryDescriptor_t src, avSize_t count);

		/**
		 * \brief This method copies memory from CUDA device to host.
		 */
		DLL_PUBLIC avStatus_t cudaCopyMemoryToHost(avContextDescriptor_t context, void *dst, const avMemoryDescriptor_t src, avSize_t bytes);

		/**
		 * \brief This method copies memory from host to CUDA device.
		 */
		DLL_PUBLIC avStatus_t cudaCopyMemoryFromHost(avContextDescriptor_t context, avMemoryDescriptor_t dst, const void *src, avSize_t bytes);

		DLL_PUBLIC avStatus_t cudaPageLock(void *ptr, avSize_t count);

		DLL_PUBLIC avStatus_t cudaPageUnlock(void *ptr);

		/**
		 * \brief Returns number of available CUDA devices.
		 * This method never fails so it returns the result directly. If anything goes wrong, it returns 0.
		 */
		DLL_PUBLIC int cudaGetNumberOfDevices();

		/**
		 * \brief Creates new tensor descriptor.
		 *
		 * \param[out] result Pointer to the new tensor descriptor.
		 */
		DLL_PUBLIC avStatus_t cudaCreateTensorDescriptor(avTensorDescriptor_t *result);

		/**
		 * \brief Deletes tensor descriptor.
		 *
		 * \param[in] desc Tensor descriptor to be deleted.
		 */
		DLL_PUBLIC avStatus_t cudaDestroyTensorDescriptor(avTensorDescriptor_t desc);

		/**
		 * \brief Sets tensor descriptor.
		 *
		 * \param[in] desc Tensor descriptor to be set.
		 * \param[in] dtype Data type of the tensor descriptor.
		 * \param[in] nbDims Number of dimensions. Must be greater than 0 and lower or equal to AVOCADO_MAX_TENSOR_DIMENSIONS.
		 * \param[in] dimensions Array with shape of the tensor. Must contain nbDims elements.
		 */
		DLL_PUBLIC avStatus_t cudaSetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t dtype, int nbDims, const int dimensions[]);

		/**
		 * \brief Queries parameters of tensor descriptor.
		 *
		 * \param[in] desc
		 * \param[out] dtype
		 * \param[out] nbDims
		 * \param[out] dimensions
		 */
		DLL_PUBLIC avStatus_t cudaGetTensorDescriptor(avTensorDescriptor_t desc, avDataType_t *dtype, int *nbDims, int dimensions[]);

		/**
		 * \brief Creates new convolution descriptor.
		 *
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t cudaCreateConvolutionDescriptor(avConvolutionDescriptor_t *result);

		/**
		 * \brief Deletes convolution descriptor.
		 *
		 * \param[in] desc
		 */
		DLL_PUBLIC avStatus_t cudaDestroyConvolutionDescriptor(avConvolutionDescriptor_t desc);

		/**
		 * \brief Sets convolution descriptor.
		 *
		 * \param[in] desc
		 * \param[in] mode
		 * \param[in] nbDims Dimensionality of the convolution. Its value must be 1, 2 or 3.
		 * \param[in] padding Array with padding offsets. This parameter is optional (can be null), a value of 0 will be used for all dimensions.
		 * \param[in] strides Array with strides. This parameter is optional (can be null), a value of 1 will be used for all dimensions.
		 * \param[in] dilation Array with dilation factors. This parameter is optional (can be null), a value of 1 will be used for all dimensions.
		 * \param[in] groups Number of groups in the convolution. Must be greaten than 0.
		 * \param[in] paddingValue Pointer to at least 16 bytes of memory with the value of tensor padding. This parameter is optional (can be null), a value of 0 will be used then.
		 */
		DLL_PUBLIC avStatus_t cudaSetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t mode, int nbDims, const int padding[],
				const int strides[], const int dilation[], int groups, const void *paddingValue);

		/**
		 * \brief Queries parameters of convolution descriptor.
		 *
		 * \param[in] desc
		 * \param[out] mode
		 * \param[out] nbDims
		 * \param[out] padding
		 * \param[out] strides
		 * \param[out] dilation
		 * \param[out] groups
		 * \param[out] paddingValue Pointer to at least 16 bytes of memory with the value of tensor padding. This parameter is optional (can be null), will be ignored then.
		 */
		DLL_PUBLIC avStatus_t cudaGetConvolutionDescriptor(avConvolutionDescriptor_t desc, avConvolutionMode_t *mode, int *nbDims, int padding[],
				int strides[], int dilation[], int *groups, void *paddingValue);

		/**
		 * \brief Creates new optimizer descriptor.
		 *
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t cudaCreateOptimizerDescriptor(avOptimizerDescriptor_t *result);

		/**
		 * \brief Deletes optimizer descriptor.
		 *
		 * \param[in] desc Optimizer descriptor to be deleted.
		 */
		DLL_PUBLIC avStatus_t cudaDestroyOptimizerDescriptor(avOptimizerDescriptor_t desc);

		/**
		 * \brief Sets optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] learningRate
		 * \param[in] useMomentum
		 * \param[in] useNesterov
		 * \param[in] beta1
		 */
		DLL_PUBLIC avStatus_t cudaSetOptimizerSGD(avOptimizerDescriptor_t desc, double learningRate, bool useMomentum, bool useNesterov,
				double beta1);

		/**
		 * \brief Queries parameters of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] learningRate
		 * \param[out] useMomentum
		 * \param[out] useNesterov
		 * \param[out] beta1
		 */
		DLL_PUBLIC avStatus_t cudaGetOptimizerSGD(avOptimizerDescriptor_t desc, double *learningRate, bool *useMomentum, bool *useNesterov,
				double *beta1);

		/**
		 * \brief Sets optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] learningRate
		 * \param[in] beta1
		 * \param[in] beta2
		 */
		DLL_PUBLIC avStatus_t cudaSetOptimizerADAM(avOptimizerDescriptor_t desc, double learningRate, double beta1, double beta2);

		/**
		 * \brief Queries parameters of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] learningRate
		 * \param[out] beta1
		 * \param[out] beta2
		 */
		DLL_PUBLIC avStatus_t cudaGetOptimizerADAM(avOptimizerDescriptor_t desc, double *learningRate, double *beta1, double *beta2);

		/**
		 * \brief Queries type of optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[out] type
		 */
		DLL_PUBLIC avStatus_t cudaGetOptimizerType(avOptimizerDescriptor_t desc, avOptimizerType_t *type);

		/**
		 * \brief This routine is used to convert between data types.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[out] dst
		 * \param[in] dstType
		 * \param[in] src
		 * \param[in] srcType
		 * \param[in] elements
		 *
		 */
		DLL_PUBLIC avStatus_t cudaChangeType(avContextDescriptor_t context, avMemoryDescriptor_t dst, avDataType_t dstType,
				const avMemoryDescriptor_t src, avDataType_t srcType, avSize_t elements);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t cudaConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] nbTensors
		 */
		DLL_PUBLIC avStatus_t cudaSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] newDimOrder
		 */
		DLL_PUBLIC avStatus_t cudaTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const int newDimOrder[]);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] alpha
		 */
		DLL_PUBLIC avStatus_t cudaScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *alpha);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] scalar
		 */
		DLL_PUBLIC avStatus_t cudaAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const void *scalar);

		/**
		 *
		 * C = alpha3 * activation(alpha1 * A + alpha2 * B) + beta1 * C
		 *
		 * Supported data type configurations:
		 *  cDesc dtype | aDesc dtype | bDesc dtype
		 * -------------|-------------|------------
		 *  INT8        | INT8        | FLOAT32
		 *  INT32       | INT8        | FLOAT32
		 *  FLOAT16     | FLOAT16     | FLOAT32
		 *  BFLOAT16    | BFLOAT16    | FLOAT32
		 *  FLOAT32     | FLOAT32     | FLOAT32
		 *  FLOAT64     | FLOAT64     | FLOAT64
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha3
		 * \param[in] alpha1
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] alpha2
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 * \param[in] activation
		 */
		DLL_PUBLIC avStatus_t cudaAddTensors(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha1
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] alpha2
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t cudaBinaryOp(avContextDescriptor_t context, avBinaryOp_t operation, const void *alpha1,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc,
				const avMemoryDescriptor_t bMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t cudaUnaryOp(avContextDescriptor_t context, avUnaryOp_t operation, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] operation
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 */
		DLL_PUBLIC avStatus_t cudaReduceTensor(avContextDescriptor_t context, avReduceOp_t operation, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *beta, const avTensorDescriptor_t cDesc,
				avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aOp
		 * \param[in] bOp
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t cudaGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] aOp
		 * \param[in] bOp
		 * \param[in] alpha
		 * \param[in] aDesc
		 * \param[in] aMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] beta
		 * \param[in] cDesc
		 * \param[out] cMem
		 * C = alpha * opA(A) opB(B) + beta * C
		 */
		DLL_PUBLIC avStatus_t cudaGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem);

		/**
		 * \brief This routine applies a specified neuron activation function element-wise over each input value.
		 * In-place operation is allowed for this routine - input and output tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha
		 * \param[in] xDesc Descriptor of input tensor.
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaActivationForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

		/**
		 * \brief This routine calculates gradient of a specified neuron activation function.
		 * In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activationDesc Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 *  dstValue = alpha * result + beta * priorDstValue
		 * \param[in] yDesc Descriptor of output tensor after the layer.
		 * \param[in] yMem
		 * \param[in] dyDesc Descriptor of gradient tensor after the layer.
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc Descriptor of gradient tensor before the layer.
		 * \param[out] dxMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaActivationBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

		/**
		 * \brief This routine applies softmax function.
		 * In-place operation is allowed for this routine - input and output tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha
		 * \param[in] xDesc Descriptor of input tensor.
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaSoftmaxForward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

		/**
		 * \brief This routine calculates gradient of the softmax function.
		 * In-place operation is allowed for this routine - gradientPrev and gradientNext tensor pointers may be equal.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activationDesc Activation descriptor. For more information, see ActivationDescriptor.
		 * \param[in] alpha, beta Scaling factors used to blend the computation result with prior value in the output layer as follows:\n
		 *  dstValue = alpha * result + beta * priorDstValue
		 * \param[in] yDesc Descriptor of output tensor after the layer.
		 * \param[in] yMem
		 * \param[in] dyDesc Descriptor of gradient tensor after the layer.
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc Descriptor of gradient tensor before the layer.
		 * \param[out] dxMem
		 *
		 * \retval AVOCADO_STATUS_SUCCESS The function launched successfully.
		 * \retval AVOCADO_STATUS_NOT_SUPPORTED The function does not support the provided configuration.
		 * \retval AVOCADO_STATUS_BAD_PARAM At least one of the following conditions are met:\n
		 * The parameter mode has an invalid enumerant value.\n
		 * The dimensions of the input tensor and output tensor differ.\n
		 * The datatype of the input tensor and output tensor differs.
		 */
		DLL_PUBLIC avStatus_t cudaSoftmaxBackward(avContextDescriptor_t context, avSoftmaxMode_t mode, const void *alpha,
				const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t cudaAffineForward(avContextDescriptor_t context, avActivationType_t activation, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] scaleBiasMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] biasMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormInference(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[in] scaleBiasMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] biasMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormForward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avTensorDescriptor_t scaleBiasMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t biasMem, avMemoryDescriptor_t meanMem, avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] activation
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] yDesc
		 * \param[in] yMem
		 * \param[in] beta
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] dyDesc
		 * \param[out] dyMem
		 * \param[in] scaleMeanVarDesc
		 * \param[in] scaleMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormBackward(avContextDescriptor_t context, avActivationType_t activation, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, const avMemoryDescriptor_t yMem,
				const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem, const avTensorDescriptor_t dyDesc,
				avMemoryDescriptor_t dyMem, const avTensorDescriptor_t scaleMeanVarDesc, const avMemoryDescriptor_t scaleMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] scaleBiasDesc
		 * \param[out] scaleUpdateMem
		 * \param[out] biasUpdateMem
		 * \param[in] meanMem
		 * \param[in] varianceMem
		 * \param[in] epsilon
		 */
		DLL_PUBLIC avStatus_t cudaBatchNormUpdate(avContextDescriptor_t context, const void *alpha, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const void *beta,
				const avTensorDescriptor_t scaleBiasDesc, avMemoryDescriptor_t scaleUpdateMem, avMemoryDescriptor_t biasUpdateMem,
				const avMemoryDescriptor_t meanMem, const avMemoryDescriptor_t varianceMem, double epsilon);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] yDesc
		 * \param[out] yMem
		 * \param[out] states
		 */
		DLL_PUBLIC avStatus_t cudaDropoutForward(avContextDescriptor_t context, const avDropoutDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t states);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 * \param[in] states
		 */
		DLL_PUBLIC avStatus_t cudaDropoutBackward(avContextDescriptor_t context, const avDropoutDescriptor_t config,
				const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem,
				const avTensorDescriptor_t states);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t cudaPoolingForward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dxDesc
		 * \param[out] dxMem
		 */
		DLL_PUBLIC avStatus_t cudaPoolingBackward(avContextDescriptor_t context, const avPoolingDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dxDesc, avMemoryDescriptor_t dxMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] filterDesc
		 * \param[in] srcDesc
		 * \param[in] srcMem
		 * \param[in] colDesc
		 * \param[out] colMem
		 */
		DLL_PUBLIC avStatus_t cudaIm2Col(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc,
				avMemoryDescriptor_t colMem);

		/**
		 * \brief Calculates required workspace size for refConvolutionBiasActivationForward.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config Convolution descriptor.
		 * \param[in] xDesc Descriptor of the input tensor.
		 * \param[in] wDesc Descriptor of the weights tensor.
		 * \param[in] bDesc Descriptor of the bias tensor.
		 * \param[out] result Pointer to the integer with number of bytes required for the workspace.
		 */
		DLL_PUBLIC avStatus_t cudaGetConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const avTensorDescriptor_t xDesc, const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, avSize_t *result);

		/**
		 * \brief Precomputes some data for future use in refConvolutionBiasActivationForward method.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config Convolution descriptor.
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] bDesc
		 * \param[in] bMem
		 * \param[out] workspace Memory descriptor for some persistent workspace.
		 */
		DLL_PUBLIC avStatus_t cudaPrecomputeConvolutionWorkspace(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				avMemoryDescriptor_t workspace);

		/**
		 * \brief Calculates convolution, adds bias and optionally some external data and applies activation function.
		 * y = activation(alpha1 * conv(x, w) + alpha2 * z + b) + beta * y
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config Convolution descriptor.
		 * \param[in] alpha1 Scaling factor of the convolution output.
		 * \param[in] xDesc Input tensor descriptor.
		 * \param[in] xMem Input memory descriptor.
		 * \param[in] wDesc Weights tensor descriptor.
		 * \param[in] wMem Weights memory descriptor.
		 * \param[in] bDesc Bias tensor descriptor.
		 * \param[in] bMem Bias memory descriptor.
		 * \param[in] activation Activation function to be applied.
		 * \param[in] alpha2 Scaling factor of the external input tensor.
		 * \param[in] zDesc External input tensor descriptor.
		 * \param[in] zMem External input memory descriptor.
		 * \param[in] beta Scaling factor of the output tensor.
		 * \param[in] yDesc Output tensor descriptor.
		 * \param[out] yMem Output memory descriptor.
		 * \param[in] workspace Memory descriptor of some persistent workspace as calculated by refPrecomputeConvolutionWorkspace method.
		 */
		DLL_PUBLIC avStatus_t cudaConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config,
				const void *alpha1, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2,
				const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const avActivationType_t activation, avMemoryDescriptor_t workspace);

		/**
		 * \brief Simplified version of the above method.
		 * y = alpha * conv(x, w) + beta * y
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] wDesc
		 * \param[in] wMem
		 * \param[in] beta
		 * \param[in] yDesc
		 * \param[out] yMem
		 */
		DLL_PUBLIC avStatus_t cudaConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config
		 * \param[in] alpha
		 * \param[in] xDesc
		 * \param[in] xMem
		 * \param[in] dyDesc
		 * \param[in] dyMem
		 * \param[in] beta
		 * \param[in] dwDesc
		 * \param[out] dwMem
		 */
		DLL_PUBLIC avStatus_t cudaConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem);

		/**
		 * \brief Computes chosen metric function, averaged over entire batch.
		 *
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] metricType Type of metric function to be calculated.
		 * \param[in] outputDesc Tensor descriptor of the output.
		 * \param[in] outputMem Memory descriptor of the output.
		 * \param[in] targetDesc Tensor descriptor of the target.
		 * \param[in] targetMem Memory descriptor of the target.
		 * \param[out] result Pointer to the floating point number.
		 */
		DLL_PUBLIC avStatus_t cudaMetricFunction(avContextDescriptor_t context, avMetricType_t metricType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] lossType
		 * \param[in] outputDesc
		 * \param[in] outputMem
		 * \param[in] targetDesc
		 * \param[in] targetMem
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t cudaLossFunction(avContextDescriptor_t context, avLossType_t lossType, const avTensorDescriptor_t outputDesc,
				const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc, const avMemoryDescriptor_t targetMem, void *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] lossType
		 * \param[in] alpha
		 * \param[in] outputDesc
		 * \param[in] outputMem
		 * \param[in] targetDesc
		 * \param[in] targetMem
		 * \param[in] beta
		 * \param[in] gradientDesc
		 * \param[out] gradientMem
		 * \param[in] isFused
		 */
		DLL_PUBLIC avStatus_t cudaLossGradient(avContextDescriptor_t context, avLossType_t lossType, const void *alpha,
				const avTensorDescriptor_t outputDesc, const avMemoryDescriptor_t outputMem, const avTensorDescriptor_t targetDesc,
				const avMemoryDescriptor_t targetMem, const void *beta, const avTensorDescriptor_t gradientDesc, avMemoryDescriptor_t gradientMem,
				bool isFused);

		/**
		 * \brief Returns number of bytes needed for the workspace of given optimizer descriptor.
		 *
		 * \param[in] desc
		 * \param[in] wDesc
		 * \param[out] result
		 */
		DLL_PUBLIC avStatus_t cudaGetOptimizerWorkspaceSize(avOptimizerDescriptor_t desc, const avTensorDescriptor_t wDesc, avSize_t *result);

		/**
		 * \param[in] context Context in which the operation is performed.
		 * \param[in] config Optimizer descriptor.
		 * \param[in] wDesc Tensor descriptor of the parameter to be updated.
		 * \param[out] wMem Memory descriptor of the parameter to be updated.
		 * \param[in] dwDesc Tensor descriptor of the gradient.
		 * \param[in] dwMem Memory descriptor of the gradient.
		 * \param[in] workspace Memory descriptor of some persistent workspace needed by the function.
		 */
		DLL_PUBLIC avStatus_t cudaOptimizerLearn(avContextDescriptor_t context, const avOptimizerDescriptor_t config,
				const avTensorDescriptor_t wDesc, avMemoryDescriptor_t wMem, const avTensorDescriptor_t dwDesc, const avTensorDescriptor_t dwMem,
				avMemoryDescriptor_t workspace);

		/**
		 * \param[in] context Context in which the operation is performed.
		 */
		DLL_PUBLIC avStatus_t cudaRegularizerL2(avContextDescriptor_t context, const avTensorDescriptor_t gradientDesc,
				avMemoryDescriptor_t gradientMem, const avTensorDescriptor_t weightDesc, const avMemoryDescriptor_t weightMem,
				const void *coefficient, const void *offset, void *loss);

#ifdef __cplusplus
		}
#endif
	} /* namespace backend */
} /* namespace avocado */

#endif /* AVOCADO_CUDA_BACKEND_H_ */

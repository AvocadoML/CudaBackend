/*
 * gemm_math.cpp
 *
 *  Created on: Sep 5, 2020
 *      Author: Maciej Kozarzewski
 */
#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cassert>

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaGemm(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			cublasOperation_t op_A = is_transpose(aOp) ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t op_B = is_transpose(bOp) ? CUBLAS_OP_T : CUBLAS_OP_N;

			int M = is_transpose(bOp) ? getTensor(bDesc).dimension(0) : getTensor(bDesc).dimension(1);
			int N = is_transpose(aOp) ? getTensor(aDesc).dimension(1) : getTensor(aDesc).dimension(0);
			int K = is_transpose(bOp) ? getTensor(bDesc).dimension(1) : getTensor(bDesc).dimension(0);

			int LDA = getTensor(aDesc).lastDim();
			int LDB = getTensor(bDesc).lastDim();
			int LDC = getTensor(cDesc).lastDim();
			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT32: // AB [int8], C [int32]
				{
					if (getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8 or getTensor(bDesc).dtype() != AVOCADO_DTYPE_INT8)
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					if (LDA % 4 != 0 or LDB % 4 != 0)
						return AVOCADO_STATUS_BAD_PARAM;

					int _alpha = getAlphaValue(alpha);
					int _beta = getBetaValue(beta);
					if ((_alpha != 0 and _alpha != 1) or (_beta != 0 and _beta != 1))
						return AVOCADO_STATUS_BAD_PARAM;

					cublasStatus_t status = cublasGemmEx(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<int8_t>(bMem), CUDA_R_8I,
							LDB, getPointer<int8_t>(aMem), CUDA_R_8I, LDA, &_beta, getPointer<int32_t>(cMem), CUDA_R_32I, LDC, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT16: // ABC [float16]
				{
					half _alpha = getAlphaValue(alpha);
					half _beta = getBetaValue(beta);
					cublasStatus_t status = cublasHgemm(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<half>(bMem), LDB,
							getPointer<half>(aMem), LDA, &_beta, getPointer<half>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT32: // ABC [float32]
				{
					float _alpha = getAlphaValue(alpha);
					float _beta = getBetaValue(beta);
					cublasStatus_t status = cublasSgemm(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<float>(bMem), LDB,
							getPointer<float>(aMem), LDA, &_beta, getPointer<float>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT64: // ABC [float64]
				{
					double _alpha = getAlphaValue<double>(alpha);
					double _beta = getBetaValue<double>(beta);
					cublasStatus_t status = cublasDgemm(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<double>(bMem), LDB,
							getPointer<double>(aMem), LDA, &_beta, getPointer<double>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX32: // ABC [complex32]
				{
					cuComplex _alpha = getAlphaValue<cuComplex>(alpha);
					cuComplex _beta = getBetaValue<cuComplex>(beta);
					cublasStatus_t status = cublasCgemm(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<cuComplex>(bMem), LDB,
							getPointer<cuComplex>(aMem), LDA, &_beta, getPointer<cuComplex>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX64: // ABC [complex64]
				{
					cuDoubleComplex _alpha = getAlphaValue<cuDoubleComplex>(alpha);
					cuDoubleComplex _beta = getBetaValue<cuDoubleComplex>(beta);
					cublasStatus_t status = cublasZgemm(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<cuDoubleComplex>(bMem), LDB,
							getPointer<cuDoubleComplex>(aMem), LDA, &_beta, getPointer<cuDoubleComplex>(cMem), LDC);
					return convertStatus(status);
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}

		avStatus_t cudaGemmBatched(avContextDescriptor_t context, avGemmOperation_t aOp, avGemmOperation_t bOp, const void *alpha,
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *beta, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			int batch = getTensor(aDesc).firstDim();
			cublasOperation_t op_A = is_transpose(aOp) ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t op_B = is_transpose(bOp) ? CUBLAS_OP_T : CUBLAS_OP_N;

			int M = is_transpose(bOp) ? getTensor(bDesc).dimension(1) : getTensor(bDesc).dimension(2);
			int N = is_transpose(aOp) ? getTensor(aDesc).dimension(2) : getTensor(aDesc).dimension(1);
			int K = is_transpose(bOp) ? getTensor(bDesc).dimension(2) : getTensor(bDesc).dimension(1);

			int LDA = getTensor(aDesc).lastDim();
			int LDB = getTensor(bDesc).lastDim();
			int LDC = getTensor(cDesc).lastDim();
			int strideA = getTensor(aDesc).volumeWithoutFirstDim();
			int strideB = getTensor(bDesc).volumeWithoutFirstDim();
			int strideC = getTensor(cDesc).volumeWithoutFirstDim();

			switch (getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
				{
					half _alpha = getAlphaValue(alpha);
					half _beta = getBetaValue(beta);
					cublasStatus_t status = cublasHgemmStridedBatched(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<half>(bMem), LDB,
							strideB, getPointer<half>(aMem), LDA, strideA, &_beta, getPointer<half>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT32:
				{
					float _alpha = getAlphaValue(alpha);
					float _beta = getBetaValue(beta);
					cublasStatus_t status = cublasSgemmStridedBatched(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<float>(bMem), LDB,
							strideB, getPointer<float>(aMem), LDA, strideA, &_beta, getPointer<float>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double _alpha = getAlphaValue<double>(alpha);
					double _beta = getBetaValue<double>(beta);
					cublasStatus_t status = cublasDgemmStridedBatched(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<double>(bMem), LDB,
							strideB, getPointer<double>(aMem), LDA, strideA, &_beta, getPointer<double>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX32:
				{
					cuComplex _alpha = getAlphaValue<cuComplex>(alpha);
					cuComplex _beta = getBetaValue<cuComplex>(beta);
					cublasStatus_t status = cublasCgemmStridedBatched(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, getPointer<cuComplex>(bMem),
							LDB, strideB, getPointer<cuComplex>(aMem), LDA, strideA, &_beta, getPointer<cuComplex>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX64:
				{
					cuDoubleComplex _alpha = getAlphaValue<cuDoubleComplex>(alpha);
					cuDoubleComplex _beta = getBetaValue<cuDoubleComplex>(beta);
					cublasStatus_t status = cublasZgemmStridedBatched(getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							getPointer<cuDoubleComplex>(bMem), LDB, strideB, getPointer<cuDoubleComplex>(aMem), LDA, strideA, &_beta,
							getPointer<cuDoubleComplex>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

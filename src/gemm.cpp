/*
 * gemm_math.cpp
 *
 *  Created on: Sep 5, 2020
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

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
			cuda::getContext(context).setDevice();
			cublasOperation_t op_A = cuda::is_transpose(aOp) ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t op_B = cuda::is_transpose(bOp) ? CUBLAS_OP_T : CUBLAS_OP_N;

			int M = cuda::is_transpose(bOp) ? cuda::getTensor(bDesc).dimension(0) : cuda::getTensor(bDesc).dimension(1);
			int N = cuda::is_transpose(aOp) ? cuda::getTensor(aDesc).dimension(1) : cuda::getTensor(aDesc).dimension(0);
			int K = cuda::is_transpose(bOp) ? cuda::getTensor(bDesc).dimension(1) : cuda::getTensor(bDesc).dimension(0);

			int LDA = cuda::getTensor(aDesc).lastDim();
			int LDB = cuda::getTensor(bDesc).lastDim();
			int LDC = cuda::getTensor(cDesc).lastDim();
			switch (cuda::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_INT32: // AB [int8], C [int32]
				{
					if (cuda::getTensor(aDesc).dtype() != AVOCADO_DTYPE_INT8 or cuda::getTensor(bDesc).dtype() != AVOCADO_DTYPE_INT8)
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
					if (LDA % 4 != 0 or LDB % 4 != 0)
						return AVOCADO_STATUS_BAD_PARAM;

					int _alpha = cuda::getAlphaValue(alpha);
					int _beta = cuda::getBetaValue(beta);
					if ((_alpha != 0 and _alpha != 1) or (_beta != 0 and _beta != 1))
						return AVOCADO_STATUS_BAD_PARAM;

					cublasStatus_t status = cublasGemmEx(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer(bMem), CUDA_R_8I,
							LDB, cuda::getPointer(aMem), CUDA_R_8I, LDA, &_beta, cuda::getPointer(cMem), CUDA_R_32I, LDC, CUDA_R_32I, CUBLAS_GEMM_DEFAULT);
					return convertStatus(status);
				}
//				case AVOCADO_DTYPE_BFLOAT16: // ABC [bfloat16]
//				{
//					float _alpha = cuda::getAlphaValue(alpha);
//					float _beta = cuda::getBetaValue(beta);
//					cublasStatus_t status = cublasGemmEx(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer(bMem),
//							CUDA_R_16BF, LDB, cuda::getPointer(aMem), CUDA_R_16BF, LDA, &_beta, cuda::getPointer(cMem), CUDA_R_16BF, LDC, CUDA_R_32F,
//							CUBLAS_GEMM_DEFAULT);
//					return convertStatus(status);
//				}
				case AVOCADO_DTYPE_FLOAT16: // ABC [float16]
				{
					int sm_ver = cuda_sm_version(cuda::getContext(context).getDevice());
					if (sm_ver == 53 or sm_ver == 60 or sm_ver >= 62)
					{
						half _alpha = cuda::getAlphaValue(alpha);
						half _beta = cuda::getBetaValue(beta);
						cublasStatus_t status = cublasHgemm(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer<half>(bMem),
								LDB, cuda::getPointer<half>(aMem), LDA, &_beta, cuda::getPointer<half>(cMem), LDC);
						return convertStatus(status);
					}
					else
					{
						float _alpha = cuda::getAlphaValue(alpha);
						float _beta = cuda::getBetaValue(beta);
						cublasStatus_t status = cublasGemmEx(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer(bMem),
								CUDA_R_16F, LDB, cuda::getPointer(aMem), CUDA_R_16F, LDA, &_beta, cuda::getPointer(cMem), CUDA_R_16F, LDC, CUDA_R_32F,
								CUBLAS_GEMM_DEFAULT);
						return convertStatus(status);
					}
				}
				case AVOCADO_DTYPE_FLOAT32: // ABC [float32]
				{
					float _alpha = cuda::getAlphaValue(alpha);
					float _beta = cuda::getBetaValue(beta);
					cublasStatus_t status = cublasSgemm(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer<float>(bMem), LDB,
							cuda::getPointer<float>(aMem), LDA, &_beta, cuda::getPointer<float>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT64: // ABC [float64]
				{
					double _alpha = cuda::getAlphaValue<double>(alpha);
					double _beta = cuda::getBetaValue<double>(beta);
					cublasStatus_t status = cublasDgemm(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer<double>(bMem),
							LDB, cuda::getPointer<double>(aMem), LDA, &_beta, cuda::getPointer<double>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX32: // ABC [complex32]
				{
					cuComplex _alpha = cuda::getAlphaValue<cuComplex>(alpha);
					cuComplex _beta = cuda::getBetaValue<cuComplex>(beta);
					cublasStatus_t status = cublasCgemm(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha, cuda::getPointer<cuComplex>(bMem),
							LDB, cuda::getPointer<cuComplex>(aMem), LDA, &_beta, cuda::getPointer<cuComplex>(cMem), LDC);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX64: // ABC [complex64]
				{
					cuDoubleComplex _alpha = cuda::getAlphaValue<cuDoubleComplex>(alpha);
					cuDoubleComplex _beta = cuda::getBetaValue<cuDoubleComplex>(beta);
					cublasStatus_t status = cublasZgemm(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							cuda::getPointer<cuDoubleComplex>(bMem), LDB, cuda::getPointer<cuDoubleComplex>(aMem), LDA, &_beta,
							cuda::getPointer<cuDoubleComplex>(cMem), LDC);
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
			cuda::getContext(context).setDevice();
			int batch = cuda::getTensor(aDesc).firstDim();
			cublasOperation_t op_A = cuda::is_transpose(aOp) ? CUBLAS_OP_T : CUBLAS_OP_N;
			cublasOperation_t op_B = cuda::is_transpose(bOp) ? CUBLAS_OP_T : CUBLAS_OP_N;

			int M = cuda::is_transpose(bOp) ? cuda::getTensor(bDesc).dimension(1) : cuda::getTensor(bDesc).dimension(2);
			int N = cuda::is_transpose(aOp) ? cuda::getTensor(aDesc).dimension(2) : cuda::getTensor(aDesc).dimension(1);
			int K = cuda::is_transpose(bOp) ? cuda::getTensor(bDesc).dimension(2) : cuda::getTensor(bDesc).dimension(1);

			int LDA = cuda::getTensor(aDesc).lastDim();
			int LDB = cuda::getTensor(bDesc).lastDim();
			int LDC = cuda::getTensor(cDesc).lastDim();
			int strideA = cuda::getTensor(aDesc).volumeWithoutFirstDim();
			int strideB = cuda::getTensor(bDesc).volumeWithoutFirstDim();
			int strideC = cuda::getTensor(cDesc).volumeWithoutFirstDim();

			switch (cuda::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
				{
					int sm_ver = cuda_sm_version(cuda::getContext(context).getDevice());
					if (sm_ver == 53 or sm_ver == 60 or sm_ver >= 62)
					{
						half _alpha = cuda::getAlphaValue(alpha);
						half _beta = cuda::getBetaValue(beta);
						cublasStatus_t status = cublasHgemmStridedBatched(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
								cuda::getPointer<half>(bMem), LDB, strideB, cuda::getPointer<half>(aMem), LDA, strideA, &_beta, cuda::getPointer<half>(cMem),
								LDC, strideC, batch);
						return convertStatus(status);
					}
					else
					{
						float _alpha = cuda::getAlphaValue(alpha);
						float _beta = cuda::getBetaValue(beta);
						cublasStatus_t status = cublasGemmStridedBatchedEx(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
								cuda::getPointer(bMem), CUDA_R_16F, LDB, strideB, cuda::getPointer(aMem), CUDA_R_16F, LDA, strideA, &_beta,
								cuda::getPointer(cMem), CUDA_R_16F, LDC, strideC, batch, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
						return convertStatus(status);
					}
				}
				case AVOCADO_DTYPE_FLOAT32:
				{
					float _alpha = cuda::getAlphaValue(alpha);
					float _beta = cuda::getBetaValue(beta);
					cublasStatus_t status = cublasSgemmStridedBatched(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							cuda::getPointer<float>(bMem), LDB, strideB, cuda::getPointer<float>(aMem), LDA, strideA, &_beta, cuda::getPointer<float>(cMem),
							LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_FLOAT64:
				{
					double _alpha = cuda::getAlphaValue<double>(alpha);
					double _beta = cuda::getBetaValue<double>(beta);
					cublasStatus_t status = cublasDgemmStridedBatched(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							cuda::getPointer<double>(bMem), LDB, strideB, cuda::getPointer<double>(aMem), LDA, strideA, &_beta, cuda::getPointer<double>(cMem),
							LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX32:
				{
					cuComplex _alpha = cuda::getAlphaValue<cuComplex>(alpha);
					cuComplex _beta = cuda::getBetaValue<cuComplex>(beta);
					cublasStatus_t status = cublasCgemmStridedBatched(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							cuda::getPointer<cuComplex>(bMem), LDB, strideB, cuda::getPointer<cuComplex>(aMem), LDA, strideA, &_beta,
							cuda::getPointer<cuComplex>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				case AVOCADO_DTYPE_COMPLEX64:
				{
					cuDoubleComplex _alpha = cuda::getAlphaValue<cuDoubleComplex>(alpha);
					cuDoubleComplex _beta = cuda::getBetaValue<cuDoubleComplex>(beta);
					cublasStatus_t status = cublasZgemmStridedBatched(cuda::getContext(context).getHandle(), op_B, op_A, M, N, K, &_alpha,
							cuda::getPointer<cuDoubleComplex>(bMem), LDB, strideB, cuda::getPointer<cuDoubleComplex>(aMem), LDA, strideA, &_beta,
							cuda::getPointer<cuDoubleComplex>(cMem), LDC, strideC, batch);
					return convertStatus(status);
				}
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
		}
	} /* namespace backend */
} /* namespace avocado */

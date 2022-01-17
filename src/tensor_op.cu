/*
 * tensor_op.cu
 *
 *  Created on: Sep 16, 2020
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	__global__ void kernel_concat_tensors(T *dst, const T *src, unsigned int first_dim, unsigned int src_last_dim, unsigned int dst_last_dim,
			unsigned int offset)
	{
		for (unsigned int i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (unsigned int j = threadIdx.x; j < src_last_dim; j += blockDim.x)
				dst[i * dst_last_dim + offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void concat_launcher(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, unsigned int offset)
	{
		const unsigned int first_dim = cuda::getTensor(cDesc).volumeWithoutLastDim();
		const unsigned int dst_last_dim = cuda::getTensor(cDesc).lastDim();
		const unsigned int src_last_dim = cuda::getTensor(aDesc).lastDim();

		dim3 gridDim(std::max(1u, std::min(512u, first_dim)));
		dim3 blockDim(std::max(1u, std::min(256u, src_last_dim)));
		kernel_concat_tensors<<<gridDim, blockDim, 0, cuda::getContext(context).getStream()>>>(cuda::getPointer<T>(cMem), cuda::getPointer<T>(aMem), first_dim, src_last_dim,
				dst_last_dim, offset);
	}

	template<typename T>
	__global__ void kernel_split_tensors(T *dst, const T *src, size_t first_dim, size_t src_last_dim, size_t dst_last_dim, size_t offset)
	{
		for (size_t i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (size_t j = threadIdx.x; j < src_last_dim; j += blockDim.x)
				dst[i * src_last_dim + j] = src[i * dst_last_dim + offset + j];
	}
	template<typename T>
	void split_launcher(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, unsigned int offset)
	{
		const unsigned int first_dim = cuda::getTensor(cDesc).volumeWithoutLastDim();
		const unsigned int dst_last_dim = cuda::getTensor(cDesc).lastDim();
		const unsigned int src_last_dim = cuda::getTensor(aDesc).lastDim();

		dim3 gridDim(std::max(1u, std::min(512u, first_dim)));
		dim3 blockDim(std::max(1u, std::min(256u, src_last_dim)));
		kernel_split_tensors<<<gridDim, blockDim, 0, cuda::getContext(context).getStream()>>>(cuda::getPointer<T>(cMem), cuda::getPointer<T>(aMem), first_dim, src_last_dim,
				dst_last_dim, offset);
	}

	template<typename T, int dim>
	__global__ void transpose(T *dst, const T *src, uint4 src_shape, uint4 ordering, unsigned int elements)
	{
		__shared__ unsigned int shape[4];
		__shared__ unsigned int order[4];

		__shared__ unsigned int src_stride[dim];
		__shared__ unsigned int dst_stride[dim];
		if (threadIdx.x == 0)
		{
			shape[0] = src_shape.x;
			shape[1] = src_shape.y;
			shape[2] = src_shape.z;
			shape[3] = src_shape.w;

			order[0] = ordering.x;
			order[1] = ordering.y;
			order[2] = ordering.z;
			order[3] = ordering.w;
			unsigned int tmp_src = 1, tmp_dst = 1;
			for (int i = dim - 1; i >= 0; i--)
			{
				src_stride[i] = tmp_src;
				dst_stride[order[i]] = tmp_dst;
				tmp_src *= shape[i];
				tmp_dst *= shape[order[i]];
			}
		}
		__syncthreads();

		for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			unsigned int tmp = i, dst_idx = 0;
			for (unsigned int j = 0; j < dim; j++)
			{
				unsigned int idx = tmp / src_stride[j];
				dst_idx += idx * dst_stride[j];
				tmp -= idx * src_stride[j];
			}
			dst[dst_idx] = src[i];
		}
	}
//	template<int dim>
//	avStatus_t transpose_launcher(mlContext_t context, mlTensor_t dst, const mlTensor_t src, const int *ordering)
//	{
//		dim3 blockDim(256);
//		dim3 gridDim(512);
	//		int4 shape { getDim(src, 0), getDim(src, 1), getDim(src, 2), getDim(src, 3) };
	//		int4 order { ordering[0], ordering[1], ordering[2], ordering[3] };
	//
	//		int elements = volume(src);
	//		switch (dataTypeSize(src->dtype))
	//		{
	//			case 1:
	//				transpose<int8_t, dim> <<<blockDim, gridDim, 0, getStream(context)>>>(data<int8_t>(dst), constData<int8_t>(src), shape, order, elements);
	//				break;
	//				case 2:
	//				transpose<int16_t, dim> <<<blockDim, gridDim, 0, getStream(context)>>>(data<int16_t>(dst), constData<int16_t>(src), shape, order, elements);
	//				break;
	//				case 4:
	//				transpose<int32_t, dim> <<<blockDim, gridDim, 0, getStream(context)>>>(data<int32_t>(dst), constData<int32_t>(src), shape, order, elements);
	//				break;
	//				case 8:
	//				transpose<int2, dim> <<<blockDim, gridDim, 0, getStream(context)>>>(data<int2>(dst), constData<int2>(src), shape, order, elements);
	//				break;
	//				case 16:
	//				transpose<int4, dim> <<<blockDim, gridDim, 0, getStream(context)>>>(data<int4>(dst), constData<int4>(src), shape, order, elements);
	//				break;
	//			}
//		return checkForErrors();
//	}

	template<typename T, typename U = T>
	__global__ void kernel_scale_tensor(T *dst, U alpha, unsigned int elements)
	{
		Store<T, U> store;
		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] = store(static_cast<U>(dst[i]) * alpha);
	}
	__global__ void kernel_scale_tensor(half *dst, float alpha, unsigned int elements)
	{
#if __CUDA_ARCH__ >= 530
		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] *= alpha;
#endif
	}

	template<typename T>
	__global__ void kernel_add_to_tensor(T *dst, T value, unsigned int elements)
	{
		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] += value;
	}
	__global__ void kernel_add_to_tensor(half *dst, half value, unsigned int elements)
	{
#if __CUDA_ARCH__ >= 530
		for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
			dst[i] += value;
#endif
	}

}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
			avSize_t last_dim_offset = 0;
			for (int i = 0; i < nbTensors; i++)
			{
				const avSize_t src_last_dim = cuda::getTensor(aDesc[i]).lastDim();
				switch (cuda::dataTypeSize(cuda::getTensor(cDesc).dtype()))
				{
					case 1:
						concat_launcher<int8_t>(context, cDesc, cMem, aDesc[i], aMem[i], last_dim_offset);
						break;
					case 2:
						concat_launcher<int16_t>(context, cDesc, cMem, aDesc[i], aMem[i], last_dim_offset);
						break;
					case 4:
						concat_launcher<int32_t>(context, cDesc, cMem, aDesc[i], aMem[i], last_dim_offset);
						break;
					case 8:
						concat_launcher<int2>(context, cDesc, cMem, aDesc[i], aMem[i], last_dim_offset);
						break;
					case 16:
						concat_launcher<int4>(context, cDesc, cMem, aDesc[i], aMem[i], last_dim_offset);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
				last_dim_offset += src_last_dim;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
			avSize_t last_dim_offset = 0;
			for (int i = 0; i < nbTensors; i++)
			{
				const avSize_t dst_last_dim = cuda::getTensor(cDesc[i]).lastDim();
				switch (cuda::dataTypeSize(cuda::getTensor(cDesc[i]).dtype()))
				{
					case 1:
						concat_launcher<int8_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 2:
						concat_launcher<int16_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 4:
						concat_launcher<int32_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 8:
						concat_launcher<int2>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 16:
						concat_launcher<int4>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
				last_dim_offset += dst_last_dim;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cudaScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *alpha)
		{
			size_t elements = cuda::getTensor(cDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(elements, blockDim.x));
			cudaStream_t stream = cuda::getContext(context).getStream();

			switch (cuda::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int64_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<half>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<bfloat16>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(cMem), cuda::getAlphaValue<double>(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<cuComplex>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getAlphaValue(alpha), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}

		avStatus_t cudaAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const void *scalar)
		{
			size_t elements = cuda::getTensor(cDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(elements, blockDim.x));
			cudaStream_t stream = cuda::getContext(context).getStream();

			switch (cuda::getTensor(cDesc).dtype())
			{
				case AVOCADO_DTYPE_UINT8:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cDesc), cuda::getScalarValue<uint8_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_INT8:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cDesc), cuda::getScalarValue<int8_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_INT16:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cDesc), cuda::getScalarValue<int16_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_INT32:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cDesc), cuda::getScalarValue<int32_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_INT64:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int64_t>(cDesc), cuda::getScalarValue<int64_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<half>(cDesc), cuda::getScalarValue<half>(scalar), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<bfloat16>(cDesc), cuda::getScalarValue<bfloat16>(scalar), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(cDesc), cuda::getScalarValue<float>(scalar), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(cDesc), cuda::getScalarValue<double>(scalar), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cDesc), cuda::getScalarValue<uint8_t>(scalar), elements);
					break;
				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cDesc), cuda::getScalarValue<uint8_t>(scalar), elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}

		avStatus_t cudaAddTensors(avContextDescriptor_t context, const void *alpha3, const void *alpha1, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const void *alpha2, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *beta,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, avActivationType_t activation)
		{
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

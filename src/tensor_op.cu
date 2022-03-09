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
#include "numbers/numbers.cuh"

#include <cstring>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace
{
	using namespace avocado::backend;

	template<typename T>
	__global__ void kernel_concat_tensors(T *dst, const T *src, uint32_t first_dim, uint32_t src_last_dim, uint32_t dst_last_dim, uint32_t offset)
	{
		for (uint32_t i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (uint32_t j = threadIdx.x; j < src_last_dim; j += blockDim.x)
				dst[i * dst_last_dim + offset + j] = src[i * src_last_dim + j];
	}
	template<typename T>
	void concat_launcher(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, uint32_t offset)
	{
		const uint32_t first_dim = cuda::getTensor(cDesc).volumeWithoutLastDim();
		const uint32_t dst_last_dim = cuda::getTensor(cDesc).lastDim();
		const uint32_t src_last_dim = cuda::getTensor(aDesc).lastDim();

		dim3 gridDim(std::max(1u, std::min(512u, first_dim)));
		dim3 blockDim(std::max(1u, std::min(256u, src_last_dim)));
		kernel_concat_tensors<<<gridDim, blockDim, 0, cuda::getContext(context).getStream()>>>(cuda::getPointer<T>(cMem), cuda::getPointer<T>(aMem), first_dim,
				src_last_dim, dst_last_dim, offset);
	}

	template<typename T>
	__global__ void kernel_split_tensors(T *dst, const T *src, uint32_t first_dim, uint32_t src_last_dim, uint32_t dst_last_dim, uint32_t offset)
	{
		for (uint32_t i = blockIdx.x; i < first_dim; i += gridDim.x)
			for (uint32_t j = threadIdx.x; j < dst_last_dim; j += blockDim.x)
				dst[i * dst_last_dim + j] = src[i * src_last_dim + offset + j];
	}
	template<typename T>
	void split_launcher(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
			const avMemoryDescriptor_t aMem, uint32_t offset)
	{
		const uint32_t first_dim = cuda::getTensor(cDesc).volumeWithoutLastDim();
		const uint32_t dst_last_dim = cuda::getTensor(cDesc).lastDim();
		const uint32_t src_last_dim = cuda::getTensor(aDesc).lastDim();

		dim3 gridDim(std::max(1u, std::min(512u, first_dim)));
		dim3 blockDim(std::max(1u, std::min(256u, src_last_dim)));
		kernel_split_tensors<<<gridDim, blockDim, 0, cuda::getContext(context).getStream()>>>(cuda::getPointer<T>(cMem), cuda::getPointer<T>(aMem), first_dim,
				src_last_dim, dst_last_dim, offset);
	}

	struct Array8D
	{
	private:
		uint32_t m_data[8];
		uint32_t m_length;
	public:
		__host__ Array8D(const cuda::TensorDescriptor &desc) :
				m_length(desc.nbDims())
		{
			for (int i = 0; i < m_length; i++)
				m_data[i] = desc.dimension(i);
		}
		__host__ Array8D(const int* data, const int length) :
				m_length(length)
		{
			for (int i = 0; i < m_length; i++)
				m_data[i] = data[i];
		}
		__device__ Array8D(uint32_t length) :
				m_length(length)
		{
		}
		__device__ uint32_t length() const
		{
			return m_length;
		}
		__device__ uint32_t operator[](uint32_t index) const
		{
			assert(index < m_length);
			return m_data[index];
		}
	};

	template<typename T>
	__global__ void kernel_transpose(T *dst, const T *src, Array8D srcShape, Array8D ordering, uint32_t elements)
	{
		__shared__ uint32_t src_stride[8];
		__shared__ uint32_t dst_stride[8];

		int dim = srcShape.length();
		if (threadIdx.x == 0)
		{
			uint32_t tmp_src = 1, tmp_dst = 1;
			for (int i = dim - 1; i >= 0; i--)
			{
				src_stride[i] = tmp_src;
				dst_stride[ordering[i]] = tmp_dst;
				tmp_src *= srcShape[i];
				tmp_dst *= srcShape[ordering[i]];
			}
		}
		__syncthreads();

		for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < elements; i += gridDim.x * blockDim.x)
		{
			uint32_t tmp = i, dst_idx = 0;
			for (uint32_t j = 0; j < dim; j++)
			{
				uint32_t idx = tmp / src_stride[j];
				dst_idx += idx * dst_stride[j];
				tmp -= idx * src_stride[j];
			}
			dst[dst_idx] = src[i];
		}
	}

	template<typename T, typename U = T>
	__global__ void kernel_scale_tensor(T *dst, const T *src, U alpha, uint32_t elements)
	{
		numbers::Number<T> _alpha(alpha);
		for (uint32_t i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * gridDim.x * blockDim.x)
		{
			numbers::Number<T> tmp = _alpha * numbers::Number<T>(src + i, elements - i);
			tmp.store(dst + i, elements - i);
		}
	}

	template<typename T>
	__global__ void kernel_add_to_tensor(T *dst, const T *src, T value, uint32_t elements)
	{
		numbers::Number<T> _value(value);
		for (uint32_t i = numbers::length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); i < elements; i += numbers::length<T>() * gridDim.x * blockDim.x)
		{
			numbers::Number<T> tmp = _value + numbers::Number<T>(src + i, elements - i);
			tmp.store(dst + i, elements - i);
		}
	}

	template<typename dstT, typename srcT, typename biasT>
	__global__ void kernel_add_bias(dstT *dst, biasT alpha1, biasT alpha2, const srcT *src, const biasT *bias, biasT beta1, biasT beta2, biasT beta3,
			const dstT *ext, uint32_t first_dim, uint32_t last_dim, avActivationType_t type)
	{
		numbers::Number<biasT> _alpha1(alpha1);
		numbers::Number<biasT> _alpha2(alpha2);
		numbers::Number<biasT> _beta1(beta1);
		numbers::Number<biasT> _beta2(beta2);
		numbers::Number<biasT> _beta3(beta3);

		for (uint32_t i = blockIdx.y; i < first_dim; i += gridDim.y)
			for (uint32_t j = numbers::length<biasT>() * (blockIdx.x * blockDim.x + threadIdx.x); j < last_dim;
					j += numbers::length<biasT>() * blockDim.x * gridDim.x)
			{
				numbers::Number<biasT> _input = _alpha2 * numbers::Number<biasT>(src + i * last_dim + j, last_dim - j);
				numbers::Number<biasT> _bias = numbers::Number<biasT>(bias + j, last_dim - j);

				numbers::Number<biasT> tmp = _input + _bias;
				if (_beta1 != numbers::zero<biasT>() or _beta2 != numbers::zero<biasT>())
				{
					numbers::Number<biasT> _ext(ext + i * last_dim + j, last_dim - j);
					tmp = _alpha1 * activation_forward(type, tmp + _beta1 * _ext) + _beta2 * _ext;
				}
				else
					tmp = _alpha1 * activation_forward(type, tmp);
				if (_beta3 != numbers::zero<biasT>())
					tmp += _beta3 * numbers::Number<biasT>(dst + i * last_dim + j, last_dim - j);
				tmp.store(dst + i * last_dim + j, last_dim - j);
			}
	}

}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaConcatTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem,
				const avTensorDescriptor_t aDesc[], const avMemoryDescriptor_t aMem[], int nbTensors)
		{
			cuda::getContext(context).setDevice();
			uint32_t last_dim_offset = 0;
			for (int i = 0; i < nbTensors; i++)
			{
				const uint32_t src_last_dim = cuda::getTensor(aDesc[i]).lastDim();
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
			return checkForErrors();
		}

		avStatus_t cudaSplitTensors(avContextDescriptor_t context, const avTensorDescriptor_t cDesc[], avMemoryDescriptor_t cMem[],
				const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, int nbTensors)
		{
			cuda::getContext(context).setDevice();
			uint32_t last_dim_offset = 0;
			for (int i = 0; i < nbTensors; i++)
			{
				const uint32_t dst_last_dim = cuda::getTensor(cDesc[i]).lastDim();
				switch (cuda::dataTypeSize(cuda::getTensor(cDesc[i]).dtype()))
				{
					case 1:
						split_launcher<int8_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 2:
						split_launcher<int16_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 4:
						split_launcher<int32_t>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 8:
						split_launcher<int2>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					case 16:
						split_launcher<int4>(context, cDesc[i], cMem[i], aDesc, aMem, last_dim_offset);
						break;
					default:
						return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
				}
				last_dim_offset += dst_last_dim;
			}
			return checkForErrors();
		}

		avStatus_t cudaTranspose(avContextDescriptor_t context, const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem, const avTensorDescriptor_t aDesc,
				const avMemoryDescriptor_t aMem, const int newDimOrder[])
		{
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();
			dim3 blockDim(256);
			dim3 gridDim(512);
			Array8D shape(cuda::getTensor(aDesc));
			Array8D order(newDimOrder, cuda::getTensor(aDesc).nbDims());

			int elements = cuda::getTensor(aDesc).volume();
			switch (cuda::dataTypeSize(cuda::getTensor(aDesc).dtype()))
			{
				case 1:
					kernel_transpose<int8_t> <<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cMem), cuda::getPointer<int8_t>(aMem), shape, order,
							elements);
					break;
				case 2:
					kernel_transpose<int16_t> <<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cMem), cuda::getPointer<int16_t>(aMem), shape, order,
							elements);
					break;
				case 4:
					kernel_transpose<int32_t> <<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cMem), cuda::getPointer<int32_t>(aMem), shape, order,
							elements);
					break;
				case 8:
					kernel_transpose<int2> <<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int2>(cMem), cuda::getPointer<int2>(aMem), shape, order,
							elements);
					break;
				case 16:
					kernel_transpose<int4> <<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int4>(cMem), cuda::getPointer<int4>(aMem), shape, order,
							elements);
					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}

		avStatus_t cudaScaleTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *alpha,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			size_t elements = cuda::getTensor(cDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(elements, blockDim.x));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(cDesc).dtype())
			{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int64_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float16>(cMem), cuda::getPointer<float16>(aMem),
							cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<bfloat16>(cMem), cuda::getPointer<bfloat16>(aMem),
							cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(cMem), cuda::getPointer<float>(aMem),
							cuda::getAlphaValue(alpha), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(cMem), cuda::getPointer<double>(aMem),
							cuda::getAlphaValue<double>(alpha), elements);
					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<cuComplex>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_scale_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getAlphaValue(alpha), elements);
//					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}

		avStatus_t cudaAddScalarToTensor(avContextDescriptor_t context, const avTensorDescriptor_t aDesc, const avMemoryDescriptor_t aMem, const void *scalar,
				const avTensorDescriptor_t cDesc, avMemoryDescriptor_t cMem)
		{
			size_t elements = cuda::getTensor(cDesc).volume();
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(elements, blockDim.x));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(cDesc).dtype())
			{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cMem), cuda::getScalarValue<int8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cMem), cuda::getScalarValue<int16_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cMem), cuda::getScalarValue<int32_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int64_t>(cMem), cuda::getScalarValue<int64_t>(scalar), elements);
//					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float16>(cMem), cuda::getPointer<float16>(aMem),
							cuda::getScalarValue<half>(scalar), elements);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<bfloat16>(cMem), cuda::getPointer<bfloat16>(aMem),
							cuda::getScalarValue<bfloat16>(scalar), elements);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(cMem), cuda::getPointer<float>(aMem),
							cuda::getScalarValue<float>(scalar), elements);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(cMem), cuda::getPointer<double>(aMem),
							cuda::getScalarValue<double>(scalar), elements);
					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}

		avStatus_t cudaAddBias(avContextDescriptor_t context, const void *alpha1, const void *alpha2, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const avTensorDescriptor_t yDesc,
				avMemoryDescriptor_t yMem, const void *beta1, const void *beta2, const void *beta3, const avMemoryDescriptor_t zMem,
				avActivationType_t activation)
		{
			cuda::BroadcastedDimensions dimensions = cuda::getBroadcastDimensions(cuda::getTensor(xDesc), cuda::getTensor(bDesc));
			dim3 blockDim(256);
			dim3 gridDim(gridSize<1024>(dimensions.first, 1), gridSize<1024>(dimensions.last, blockDim.x));
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(yDesc).dtype())
			{
//				case AVOCADO_DTYPE_UINT8:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT8:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int8_t>(cMem), cuda::getScalarValue<int8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT16:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int16_t>(cMem), cuda::getScalarValue<int16_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT32:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int32_t>(cMem), cuda::getScalarValue<int32_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_INT64:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<int64_t>(cMem), cuda::getScalarValue<int64_t>(scalar), elements);
//					break;
				case AVOCADO_DTYPE_FLOAT16:
					kernel_add_bias<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float16>(yMem), cuda::getAlphaValue(alpha1), cuda::getAlphaValue(alpha2),
							cuda::getPointer<float16>(xMem), cuda::getPointer<float>(bMem), cuda::getBetaValue(beta1), cuda::getBetaValue(beta2),
							cuda::getBetaValue(beta3), cuda::getPointer<float16>(zMem), dimensions.first, dimensions.last, activation);
					break;
				case AVOCADO_DTYPE_BFLOAT16:
					kernel_add_bias<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<bfloat16>(yMem), cuda::getAlphaValue(alpha1),
							cuda::getAlphaValue(alpha2), cuda::getPointer<bfloat16>(xMem), cuda::getPointer<float>(bMem), cuda::getBetaValue(beta1),
							cuda::getBetaValue(beta2), cuda::getBetaValue(beta3), cuda::getPointer<bfloat16>(zMem), dimensions.first, dimensions.last,
							activation);
					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_add_bias<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<float>(yMem), cuda::getAlphaValue(alpha1), cuda::getAlphaValue(alpha2),
							cuda::getPointer<float>(xMem), cuda::getPointer<float>(bMem), cuda::getBetaValue(beta1), cuda::getBetaValue(beta2),
							cuda::getBetaValue(beta3), cuda::getPointer<float>(zMem), dimensions.first, dimensions.last, activation);
					break;
				case AVOCADO_DTYPE_FLOAT64:
					kernel_add_bias<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<double>(yMem), cuda::getAlphaValue<double>(alpha1),
							cuda::getAlphaValue<double>(alpha2), cuda::getPointer<double>(xMem), cuda::getPointer<double>(bMem),
							cuda::getBetaValue<double>(beta1), cuda::getBetaValue<double>(beta2), cuda::getBetaValue<double>(beta3),
							cuda::getPointer<double>(zMem), dimensions.first, dimensions.last, activation);
					break;
//				case AVOCADO_DTYPE_COMPLEX32:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
//				case AVOCADO_DTYPE_COMPLEX64:
//					kernel_add_to_tensor<<<gridDim, blockDim, 0, stream>>>(cuda::getPointer<uint8_t>(cMem), cuda::getScalarValue<uint8_t>(scalar), elements);
//					break;
				default:
					return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
			}
			return checkForErrors();
		}
	} /* namespace backend */
} /* namespace avocado */

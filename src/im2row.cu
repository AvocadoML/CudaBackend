/*
 * im2row.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>

namespace
{
	__device__ int remainder(int &number, int divisor)
	{
		int tmp = number / divisor;
		int result = number - divisor * tmp;
		number = tmp;
		return result;
	}

	template<typename T>
	__launch_bounds__(256, 8)
	__global__ void kernel_im2row_conv2d(const T *input, TensorShape inputShape, T *matrix, TensorShape outputShape, int2 kernelSize, int2 padding, bool invert,
			T paddingValue)
	{
		int ext_input_height = inputShape.height - 2 * padding.x;
		int ext_input_width = inputShape.width - 2 * padding.y;

		int volume = inputShape.batch * ext_input_height * ext_input_width * inputShape.filters;
		for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < volume; tid += gridDim.x * blockDim.x)
		{
			int tmp = tid;
			int in_f = remainder(tmp, inputShape.filters);
			int in_w = remainder(tmp, ext_input_width) + padding.y;
			int in_h = remainder(tmp, ext_input_height) + padding.x;
			int in_b = remainder(tmp, inputShape.batch);

			T value = paddingValue;
			if (in_h >= 0 and in_h < inputShape.height and in_w >= 0 and in_w < inputShape.width)
				value = input[inputShape.offset_at(in_b, in_h, in_w, in_f)];

			for (int i = 0; i < kernelSize.x; i++)
				for (int j = 0; j < kernelSize.y; j++)
				{
					int x = in_h + i - kernelSize.x / 2;
					int y = in_w + j - kernelSize.y / 2;
					int offset = i * kernelSize.y + j;
					if (invert == false)
						offset = (kernelSize.x * kernelSize.y - 1) - offset;
					if (x >= 0 and x < outputShape.height and y >= 0 and y < outputShape.width)
					{
						int tile_idx = in_b * outputShape.height * outputShape.width + x * outputShape.width + y;
						int asdf = (tile_idx * kernelSize.x * kernelSize.y + offset) * inputShape.filters + in_f;
						matrix[asdf] = value;
					}
				}
		}
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		bool is_conv(int expectedSize, const TensorDescriptor &wDesc) noexcept
		{
			for (int i = 0; i < wDesc.nbDims() - 2; i++)
				if (wDesc.dimension(1 + i) != expectedSize)
					return false;
			return true;
		}

		avStatus_t cudaIm2Row(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t rowDesc, avMemoryDescriptor_t rowMem)
		{
			getContext(context).setDevice();
			const bool invert = (getConvolution(config).mode == AVOCADO_CROSS_CORRELATION_MODE);
			TensorShape input_shape(getTensor(srcDesc));
			TensorShape output_shape(getConvolution(config).getOutputShape(getTensor(srcDesc), getTensor(filterDesc)));

			const int2 kernel_size = { getTensor(filterDesc).dimension(1), getTensor(filterDesc).dimension(2) };
			const int2 padding = { getConvolution(config).padding[0], getConvolution(config).padding[1] };

			dim3 blockSize(256);
			dim3 gridSize(std::min(2048, (getTensor(srcDesc).volume() + 255) / 256));
			cudaStream_t stream = getContext(context).getStream();

#define KERNEL_LAUNCH(type)\
				{type padding_value = getConvolution(config).getPaddingValue<type>();\
				kernel_im2row_conv2d<<<gridSize, blockSize, 0, stream>>>(getPointer<type>(srcMem), input_shape, getPointer<type>(rowMem), output_shape, kernel_size, padding, invert, padding_value);\
				break;}

			switch (dataTypeSize(getTensor(srcDesc).dtype()))
			{
				default:
				case 1:
					KERNEL_LAUNCH(char)
				case 2:
					KERNEL_LAUNCH(short)
				case 4:
					KERNEL_LAUNCH(int)
				case 8:
					KERNEL_LAUNCH(int2)
				case 16:
					KERNEL_LAUNCH(int4)
			}
#undef KERNEL_LAUNCH
			return checkForErrors();
		}

	} /* namespace backend */
} /* namespace avocado */

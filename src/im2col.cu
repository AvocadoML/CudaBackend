/*
 * im2col.cu
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
	int get_block_size(int size)
	{
		int tmp[] = { 1, 2, 4 };
		int result = 1;
		for (int i = 0; i < 3; i++)
			if (size % tmp[i] == 0)
				result = std::max(result, tmp[i]);
		return result;
	}

	template<typename T>
	__launch_bounds__(256, 8)
	__global__ void kernel_im2col(const T *input, T *matrix, uint4 input_shape, unsigned int kernel_size, bool invert)
	{
		unsigned int input_height = input_shape.y + kernel_size - 1;
		unsigned int input_width = input_shape.z + kernel_size - 1;
		unsigned int filters = input_shape.w;

		unsigned int volume = input_shape.x * input_height * input_width * filters;
		for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < volume; tid += gridDim.x * blockDim.x)
		{
			int in_f = tid;
			int in_b = in_f / (input_height * input_width * filters);
			in_f -= in_b * input_height * input_width * filters;
			int in_h = in_f / (input_width * filters);
			in_f -= in_h * input_width * filters;
			int in_w = in_f / filters;
			in_f -= in_w * filters;

			in_h = in_h - kernel_size / 2;
			in_w = in_w - kernel_size / 2;

			T loaded = static_cast<T>(0);
			int idx = ((in_b * input_shape.y + in_h) * input_shape.z + in_w) * filters + in_f;
			if (in_h >= 0 && in_h < input_shape.y && in_w >= 0 && in_w < input_shape.z)
				loaded = input[idx];

			for (int i = 0; i < kernel_size; i++)
				for (int j = 0; j < kernel_size; j++)
				{
					int x = in_h + i - kernel_size / 2;
					int y = in_w + j - kernel_size / 2;
					int offset = i * kernel_size + j;
					if (invert == false)
						offset = (kernel_size * kernel_size - 1) - offset;
					if (x >= 0 && x < input_shape.y && y >= 0 && y < input_shape.z)
						matrix[(((in_b * input_shape.y + x) * input_shape.z + y) * kernel_size * kernel_size + offset) * filters + in_f] = loaded;
				}
		}
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cudaIm2Col(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t filterDesc,
				const avTensorDescriptor_t srcDesc, const avMemoryDescriptor_t srcMem, const avTensorDescriptor_t colDesc, avMemoryDescriptor_t colMem)
		{
//			int last_dim = lastDim(input) * dataTypeSize(input->dtype);
//			int4 input_shape { getDim(input, 0), getDim(input, 1), getDim(input, 2), last_dim };
//			dim3 blockSize(256);
//			dim3 gridSize(std::min(2048, (volume(input) + 255) / 256));
//
//#define KERNEL_LAUNCH(type)\
//				input_shape.w /= sizeof(type);\
//				gpu_conv2D_receptive_fields<type> <<<gridSize, blockSize, 0, getStream(context)>>>( data<type>(input), data<type>(output), input_shape, kernel_height, invert);\
//				break;
//
//			switch (get_block_size(last_dim))
//			{
//				default:
//				case 1:
//					KERNEL_LAUNCH(char)
//				case 2:
//					KERNEL_LAUNCH(short)
//				case 4:
//					KERNEL_LAUNCH(int)
//			}
//#undef KERNEL_LAUNCH
//			return cudaGetLastError();
//			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

/*
 * conv2d_implicit_gemm.cu
 *
 *  Created on: Jan 02, 2022
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <array>

namespace
{
	using namespace avocado::backend;

	struct TensorShape
	{
		int batch, height, width, filters;

		__device__ int offset_at(int b, int h, int w, int f) const noexcept
		{
			assert(b >= 0 && b < batch);
			assert(h >= 0 && h < height);
			assert(w >= 0 && w < width);
			assert(f >= 0 && f < filters);
			return ((b * height + h) * width + w) * filters + f;
		}
		template<int TileSize>
		__device__ int tile_index(int b, int h, int w) const noexcept
		{
			assert(b >= 0 && b < batch);
			assert(h >= 0 && h < tiles_vertically<TileSize>());
			assert(w >= 0 && w < tiles_horizontally<TileSize>());
			return (b * tiles_vertically<TileSize>() + h) * tiles_horizontally<TileSize>() + w;
		}
		template<int TileSize>
		__device__ int tiles_vertically() const noexcept
		{
			return (height + TileSize - 1) / TileSize;
		}
		template<int TileSize>
		__device__ int tiles_horizontally() const noexcept
		{
			return (width + TileSize - 1) / TileSize;
		}
	};

	template<typename T>
	struct Line
	{
		T x0, x1, x2, x3;

		__device__ Line()
		{
		}
		__device__ Line(T value)
		{
			x0 = value;
			x1 = value;
			x2 = value;
			x3 = value;
		}
		__device__ Line(const T* src, const int stride)
		{
			x0 = src[0 * stride];
			x1 = src[1 * stride];
			x2 = src[2 * stride];
			x3 = src[3 * stride];
		}
	};
	template<typename T>
	struct Tile
	{
		T x00, x01, x02, x03;
		T x10, x11, x12, x13;
		T x20, x21, x22, x23;
		T x30, x31, x32, x33;

		__device__ Tile()
		{
		}
		__device__ Tile(T value)
		{
			x00 = value;
			x01 = value;
			x02 = value;
			x03 = value;

			x10 = value;
			x11 = value;
			x12 = value;
			x13 = value;

			x20 = value;
			x21 = value;
			x22 = value;
			x23 = value;

			x30 = value;
			x31 = value;
			x32 = value;
			x33 = value;
		}
		__device__ void print() const noexcept
		{
			printf("%i %i %i %i\n", x00, x01, x02, x03);
			printf("%i %i %i %i\n", x10, x11, x12, x13);
			printf("%i %i %i %i\n", x20, x21, x22, x23);
			printf("%i %i %i %i\n", x30, x31, x32, x33);
			printf("\n");
		}
	};

	template<typename T>
	__device__ void fma(Tile<T> &result, const Line<T> &lhs, const Line<T> &rhs) noexcept
	{
		result.x00 += lhs.x0 * rhs.x0;
		result.x01 += lhs.x0 * rhs.x1;
		result.x02 += lhs.x0 * rhs.x2;
		result.x03 += lhs.x0 * rhs.x3;

		result.x10 += lhs.x1 * rhs.x0;
		result.x11 += lhs.x1 * rhs.x1;
		result.x12 += lhs.x1 * rhs.x2;
		result.x13 += lhs.x1 * rhs.x3;

		result.x20 += lhs.x2 * rhs.x0;
		result.x21 += lhs.x2 * rhs.x1;
		result.x22 += lhs.x2 * rhs.x2;
		result.x23 += lhs.x2 * rhs.x3;

		result.x30 += lhs.x3 * rhs.x0;
		result.x31 += lhs.x3 * rhs.x1;
		result.x32 += lhs.x3 * rhs.x2;
		result.x33 += lhs.x3 * rhs.x3;
	}
	__device__ void fma(Tile<int> &result, const Line<int> &lhs, const Line<int> &rhs) noexcept
	{
#if __CUDA_ARCH__ >= 610
		result.x00 = __dp4a(lhs.x0, rhs.x0, result.x00);
		result.x01 = __dp4a(lhs.x0, rhs.x1, result.x01);
		result.x02 = __dp4a(lhs.x0, rhs.x2, result.x02);
		result.x03 = __dp4a(lhs.x0, rhs.x3, result.x03);

		result.x10 = __dp4a(lhs.x1, rhs.x0, result.x10);
		result.x11 = __dp4a(lhs.x1, rhs.x1, result.x11);
		result.x12 = __dp4a(lhs.x1, rhs.x2, result.x12);
		result.x13 = __dp4a(lhs.x1, rhs.x3, result.x13);

		result.x20 = __dp4a(lhs.x2, rhs.x0, result.x20);
		result.x21 = __dp4a(lhs.x2, rhs.x1, result.x21);
		result.x22 = __dp4a(lhs.x2, rhs.x2, result.x22);
		result.x23 = __dp4a(lhs.x2, rhs.x3, result.x23);

		result.x30 = __dp4a(lhs.x3, rhs.x0, result.x30);
		result.x31 = __dp4a(lhs.x3, rhs.x1, result.x31);
		result.x32 = __dp4a(lhs.x3, rhs.x2, result.x32);
		result.x33 = __dp4a(lhs.x3, rhs.x3, result.x33);
#endif
	}

	__device__ int2 split_thread_index(const int dimx) noexcept
	{
		return int2 { static_cast<int>(threadIdx.x) % dimx, static_cast<int>(threadIdx.x) / dimx };
	}
	__device__ int2 split_block_dim(const int dimx) noexcept
	{
		assert(static_cast<int>(blockDim.x) % dimx == 0);
		return int2 { dimx, static_cast<int>(blockDim.x) / dimx };
	}

	template<typename InputType, class Activation, int KernelSize = 3, typename ScaleType = InputType, typename OutputType = InputType>
	__launch_bounds__(256, 2)
	__global__ void kernel_conv_implicit_gemm(const InputType* weights, const InputType* input, TensorShape input_shape, OutputType* output,
			TensorShape output_shape, const int2 padding, ScaleType alpha1, ScaleType alpha2, ScaleType beta, const ScaleType* bias = nullptr,
			const OutputType* add = nullptr)
	{
		assert(blockDim.x == 256);
		constexpr int OutputTileSize = 8;
		constexpr int InputTileSize = OutputTileSize + KernelSize - 1;
		constexpr int OutputFilterFragments = 128;
		constexpr int InputFilterFragments = 32;

		__shared__ InputType input_storage[InputTileSize * InputTileSize * InputFilterFragments];
		__shared__ InputType weight_storage[OutputFilterFragments * InputFilterFragments];

		Tile<InputType> acc0(zero<InputType>());
		Tile<InputType> acc1(zero<InputType>());
		for (int input_filter = 0; input_filter < input_shape.filters; input_filter += InputFilterFragments)
		{
			int2 tmp_thread_idx = split_thread_index(InputFilterFragments); // divide into 32x8
			int2 tmp_block_dim = split_block_dim(InputFilterFragments); // divide into 32x8

			int tmp = (output_shape.filters + OutputFilterFragments - 1) / OutputFilterFragments;
			int batch = blockIdx.x / tmp;

			// input_storage [InputTileSize x InputTileSize x InputFilterFragments]
			for (int i = tmp_thread_idx.y; i < square(InputTileSize); i += tmp_block_dim.y)
			{
				int row = i / InputTileSize;
				int col = i % InputTileSize;
				int x = padding.x + OutputTileSize * blockIdx.y + row;
				int y = padding.y + OutputTileSize * blockIdx.z + col;
//				if (tmp_tid_x == 0 and input_filter == 0)
//					printf("%i - (%i, %i) , (%i, %i)\n", i, row, col, x, y);

				int index = (row * InputTileSize + col) * InputFilterFragments + tmp_thread_idx.x;
				if (x >= 0 and x < input_shape.height and y >= 0 and y < input_shape.width)
				{
					int offset = input_shape.offset_at(batch, x, y, input_filter + tmp_thread_idx.x);
//					if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and input_filter == 0)
//						printf("(%i, %i) = %i\n", row, col, input[offset]);
					input_storage[index] = input[offset];
				}
				else
					input_storage[index] = zero<InputType>();
			}

//			__syncthreads();
//			if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and input_filter == 0)
//			{
//				for (int i = 0; i < InputTileSize; i++)
//				{
//					for (int j = 0; j < InputTileSize; j++)
//					{
//						int x = input_storage[(i * InputTileSize + j) * InputFilterFragments + 10];
//						printf("%i ", x);
//					}
//					printf("\n");
//				}
//			}
//			__syncthreads();

			for (int w = 0; w < KernelSize * KernelSize; w++)
			{
				// weight_storage [OutputFilterFragments x InputFilterFragments]
//				int output_filter = (blockIdx.x % tmp) * OutputFilterFragments;
//				for (int out = tmp_thread_idx.y; out < OutputFilterFragments; out += tmp_block_dim.y)
//				{
//					int offset = ((output_filter + out) * KernelSize * KernelSize + w) * input_shape.filters + input_filter + tmp_thread_idx.x;
//					int index = out * InputFilterFragments + tmp_thread_idx.x;
//					weight_storage[index] = weights[offset];
//				}
//
//				__syncthreads();
//				if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and input_filter == 0)
//					printf("weight = %i\n\n", weight_storage[9 * InputFilterFragments + 0]);
//				__syncthreads();
				for (int k = 0; k < InputFilterFragments; k++)
				{
					Line<InputType> weight_line(weight_storage + 0, 0);
					Line<InputType> input_line0(input_storage + 0, 10);
					Line<InputType> input_line1(input_storage + 1024, 10);

					fma(acc0, input_line0, weight_line);
					fma(acc1, input_line1, weight_line);
				}
				__syncthreads();
			}
		}
//		if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0)
//		{
//			acc0.print();
//			acc1.print();
//		}
		int tmp = (output_shape.filters + OutputFilterFragments - 1) / OutputFilterFragments;
		int batch = blockIdx.x / tmp;
		int output_filter = (blockIdx.x % tmp) * OutputFilterFragments;

		__shared__ ScaleType bias_storage[OutputFilterFragments];
		if (bias != nullptr and threadIdx.x < OutputFilterFragments)
			bias_storage[threadIdx.x] = bias[output_filter + threadIdx.x];
		__syncthreads();

//		int2 tmp_thread_idx = split_thread_index(OutputFilterFragments); // divide into 64x4
//		int2 tmp_block_dim = split_block_dim(OutputFilterFragments); // divide into 64x4
//		for (int i = 0; i < 2; i++)
//			for (int j = 0; j < 2; j++)
//			{
//				int x = 2 * (tmp_thread_idx.y / 2) + i;
//				int y = 2 * (tmp_thread_idx.y % 2) + j;
//				int tmp = OutputTileSize / 2;
//				weight_storage[(x * OutputTileSize + y) * OutputFilterFragments + tmp_thread_idx.x] = acc00.at(i, j);
//				weight_storage[(x * OutputTileSize + y + tmp) * OutputFilterFragments + tmp_thread_idx.x] = acc01.at(i, j);
//				weight_storage[((x + tmp) * OutputTileSize + y) * OutputFilterFragments + tmp_thread_idx.x] = acc10.at(i, j);
//				weight_storage[((x + tmp) * OutputTileSize + y + tmp) * OutputFilterFragments + tmp_thread_idx.x] = acc11.at(i, j);
//			}

//		__syncthreads();
		int2 tmp_thread_idx = split_thread_index(InputFilterFragments); // divide into 32x8
		int2 tmp_block_dim = split_block_dim(InputFilterFragments); // divide into 32x8

		for (int row = tmp_thread_idx.y; row < OutputTileSize; row += tmp_block_dim.y)
			for (int col = 0; col < OutputTileSize; col++)
			{
				int x = OutputTileSize * blockIdx.y + row;
				int y = OutputTileSize * blockIdx.z + col;
				if (x < output_shape.height and y < output_shape.width)
				{
					ScaleType tmp = alpha1 * (acc0.x00 + acc1.x00) / 2;

					int index = output_shape.offset_at(batch, x, y, output_filter);
					if (add != nullptr)
						tmp += alpha2 * add[index];
					tmp = Activation().forward(tmp);
					if (beta != zero<ScaleType>())
						tmp += beta * output[index];
					output[index + tmp_thread_idx.x] = tmp;
				}
			}
	}

	TensorShape get_tensor_shape(const cuda::TensorDescriptor &desc)
	{
		return TensorShape( { desc.dimension(0), desc.dimension(1), desc.dimension(2), desc.dimension(3) });
	}
}

namespace avocado
{
	namespace backend
	{
		avStatus_t cuda_convolution2dImplicitGemm(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
			TensorShape input_shape = get_tensor_shape(cuda::getTensor(xDesc));
			TensorShape output_shape = get_tensor_shape(cuda::getTensor(yDesc));

			input_shape.filters /= 4;

			int batch_size = cuda::getTensor(xDesc).dimension(0);
			int tile_h = (cuda::getTensor(xDesc).dimension(1) + 7) / 8;
			int tile_w = (cuda::getTensor(xDesc).dimension(2) + 7) / 8;
			int filters_in = input_shape.filters;
			int filters_out = output_shape.filters;

			dim3 blockDim(256);
			dim3 gridDim(batch_size * ((filters_out + 127) / 128), tile_h, tile_w);
//			dim3 gridDim(1, 1, 1);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			int2 padding { -1, -1 };

			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_INT8:
				{
					float _alpha1 = cuda::getAlphaValue(alpha1);
					float _alpha2 = cuda::getAlphaValue(alpha2);
					float _beta = cuda::getBetaValue(beta);
					kernel_conv_implicit_gemm<int, ActivationLinear<float>, 3, float, int> <<<gridDim, blockDim, 0, stream>>>( cuda::getPointer<int>(wMem),
							 cuda::getPointer<int>(xMem), input_shape,  cuda::getPointer<int>(yMem), output_shape, padding, _alpha1, _alpha2, _beta,  cuda::getPointer<float>(bMem),
							 cuda::getPointer<int>(zMem));
					break;
				}
				case AVOCADO_DTYPE_FLOAT32:
				{
					float _alpha1 = cuda::getAlphaValue(alpha1);
					float _alpha2 = cuda::getAlphaValue(alpha2);
					float _beta = cuda::getBetaValue(beta);
					kernel_conv_implicit_gemm<float, ActivationLinear<float>, 3> <<<gridDim, blockDim, 0, stream>>>( cuda::getPointer<float>(wMem),
							 cuda::getPointer<float>(xMem), input_shape,  cuda::getPointer<float>(yMem), output_shape, padding, _alpha1, _alpha2, _beta,
							 cuda::getPointer<float>(bMem),  cuda::getPointer<float>(zMem));
					break;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

/*
 * convolution.cu
 *
 *  Created on: Dec 27, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>

namespace
{
	using namespace avocado::backend;

#define tensorIdx4D(b,h,w,f) ((((b)*(height)+h)*(width)+w)*(filters)+f)
#define tensorIdx3D(b,p,f, plane, filters) (((b)*(plane)+p)*(filters)+f)
#define tensorIdx2D(h,w, width) ((h)*(width)+w)

	template<unsigned int tileSize, unsigned int kernelSize, typename T>
	struct WeightTransform
	{
		__device__ void loadFrom(const T* src, unsigned int offset, unsigned int stride) noexcept
		{
		}
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
		}
	};
	template<typename T>
	struct WeightTransform<4, 3, T>
	{
	private:
		T load0, load1, load2;
	public:
		__device__ void loadFrom(const T* src, unsigned int offset, unsigned int stride) noexcept
		{
			load0 = src[offset + 0 * stride];
			load1 = src[offset + 1 * stride];
			load2 = src[offset + 2 * stride];
		}
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
			T c23 = static_cast<T>(2.0 / 3.0);
			T c13 = static_cast<T>(1.0 / 3.0);
			T c2 = static_cast<T>(2);
			T c4 = static_cast<T>(4);
			dst[offset + 0 * stride] = load0;
			dst[offset + 1 * stride] = c23 * (load0 + load1 + load2);
			dst[offset + 2 * stride] = c23 * (load0 - load1 + load2);
			dst[offset + 3 * stride] = c13 * (load0 + c2 * load1 + c4 * load2);
			dst[offset + 4 * stride] = c13 * (load0 - c2 * load1 + c4 * load2);
			dst[offset + 5 * stride] = c2 * load2;
		}
	};

#define weightTensorIndex(out, h, w, in) ((((out) * kernelSize + (h)) * kernelSize + (w)) * filters_in + (in))
#define weightMatrixIndex(m, out, in) (((m) * filters_out + (out)) * filters_in + (in))

	template<unsigned int winogradTileSize, unsigned int kernelSize, unsigned int elements, typename T, typename U = T>
	__global__ void kernel_winograd_weight_transform(T* matrices, const T* weights, unsigned int filters_out, unsigned int filters_in, bool invert)
	{
		assert(blockDim.x == elements && blockDim.y == 1);
		constexpr unsigned int tile_size = winogradTileSize + kernelSize - 1;
		__shared__ U storage[kernelSize * tile_size * elements];

		for (unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; j < filters_in; j += gridDim.x * blockDim.x)
		{
			WeightTransform<winogradTileSize, kernelSize, U> tile;
			for (unsigned int i = 0; i < kernelSize; i++)
			{
				unsigned int offset, stride;
				if (invert)
				{
					offset = weightTensorIndex(blockIdx.y, kernelSize - 1, kernelSize - 1 - i, j);
					stride = -kernelSize * filters_in;
				}
				else
				{
					offset = weightTensorIndex(blockIdx.y, 0, i, j);
					stride = kernelSize * filters_in;
				}
				tile.loadFrom(weights, offset, stride);

				offset = i * elements + threadIdx.x;
				stride = kernelSize * elements;
				tile.transformInto(storage, offset, stride);
			}

			for (unsigned int i = 0; i < tile_size; i++)
			{
				unsigned int offset = i * kernelSize * elements + threadIdx.x;
				unsigned int stride = elements;
				tile.loadFrom(storage, offset, stride);

				offset = weightMatrixIndex(tile_size * i, blockIdx.y, j);
				stride = filters_out * filters_in;
				tile.transformInto(matrices, offset, stride);
			}
		}
	}

#undef weightTensorIndex
#undef weightMatrixIndex

	template<unsigned int tileSize, unsigned int kernelSize, typename T>
	struct InputTransform
	{
	private:
		T load[tileSize + kernelSize - 1];
	public:
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
		}
		__device__ T& operator[](unsigned int idx) noexcept
		{
			assert(idx < (tileSize + kernelSize - 1));
			return load[idx];
		}
	};
	template<typename T>
	struct InputTransform<4, 3, T>
	{
	private:
		T load[6];
	public:
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
			T c025 = static_cast<T>(0.25);
			T c05 = static_cast<T>(0.5);

			dst[offset + 0 * stride] = load[0] - load[2] + c025 * (load[4] - load[2]);
			dst[offset + 1 * stride] = load[1] + load[2] - c025 * (load[3] + load[4]);
			dst[offset + 2 * stride] = load[2] - load[1] + c025 * (load[3] - load[4]);
			dst[offset + 3 * stride] = load[3] - load[1] + c05 * (load[4] - load[2]);
			dst[offset + 4 * stride] = load[1] - load[3] + c05 * (load[4] - load[2]);
			dst[offset + 5 * stride] = load[1] - load[3] + c025 * (load[5] - load[3]);
		}
		__device__ T& operator[](unsigned int idx) noexcept
		{
			assert(idx < 6);
			return load[idx];
		}
	};

#define inputTensorIndex(b, h, w, f) ((((b) * shape.y + (h)) * shape.z + (w)) * shape.w + (f))

	template<unsigned int winogradTileSize, unsigned int kernelSize, unsigned int elements, typename T, typename U = T>
	__global__ void kernel_winograd_input_transform(T* matrices, const T* input, uint4 shape, int2 padding, T padding_value = zero<T>())
	{
		assert(blockDim.x == elements && blockDim.y == 1);
		constexpr unsigned int tile_size = winogradTileSize + kernelSize - 1;
		__shared__ U storage[tile_size * tile_size * elements];

		for (unsigned int f = threadIdx.x; f < shape.w; f += elements)
		{
			InputTransform<winogradTileSize, kernelSize, U> tile;
			for (int w = 0; w < tile_size; w++)
			{
				for (int h = 0; h < tile_size; h++)
				{
					int b = blockIdx.x;
					int x = padding.x + winogradTileSize * blockIdx.y + h;
					int y = padding.y + winogradTileSize * blockIdx.z + w;
//					printf("%i %i %i %i\n", b, x, y, f);
					if (x >= 0 and x < shape.y and y >= 0 and y < shape.z)
						tile[h] = input[inputTensorIndex(b, x, y, f)];
					else
						tile[h] = padding_value;
				}

//				for (int h = 0; h < tile_size; h++)
//					printf("%f ", tile[h]);
//				printf("\n");
				unsigned int offset = w * elements + threadIdx.x;
				unsigned int stride = tile_size * elements;
				tile.transformInto(storage, offset, stride);
			}

//			printf("---------------------------------\n");
//			for (int w = 0; w < tile_size; w++)
//			{
//				for (int h = 0; h < tile_size; h++)
//					printf("%f ", storage[(w * 6 + h) * elements]);
//				printf("\n");
//			}
//			printf("---------------------------------\n");

			for (int w = 0; w < tile_size; w++)
			{
				unsigned int offset = w * tile_size * elements + threadIdx.x;
				unsigned int stride = elements;
				for (int h = 0; h < tile_size; h++)
					tile[h] = storage[offset + h * stride];

//				for (int h = 0; h < tile_size; h++)
//					printf("%f ", tile[h]);
//				printf("\n");

				stride = gridDim.x * gridDim.y * gridDim.z * shape.w;
				offset = tile_size * w * stride + ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.w + f;
//				printf("%i %i\n", offset, stride);
				tile.transformInto(matrices, offset, stride);
			}
		}
	}
#undef inputTensorIndex

	template<unsigned int tileSize, unsigned int kernelSize, typename T>
	struct OutputTransform
	{
	private:
		T load[tileSize + kernelSize - 1];
	public:
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
		}
		__device__ T& operator[](unsigned int idx) noexcept
		{
			assert(idx < (tileSize + kernelSize - 1));
			return load[idx];
		}
	};
	template<typename T>
	struct OutputTransform<4, 3, T>
	{
	private:
		T load[6];
	public:
		__device__ void transformInto(T* dst, unsigned int offset, unsigned int stride) const noexcept
		{
			T c025 = static_cast<T>(0.25);
			T c05 = static_cast<T>(0.5);
			T c2 = static_cast<T>(2.0);

			dst[offset + 0 * stride] = load[0] + load[1] + load[2] + c025 * (load[3] + load[4]);
			dst[offset + 1 * stride] = load[1] - load[2] + c05 * (load[3] - load[4]);
			dst[offset + 2 * stride] = load[1] + load[2] + load[3] + load[4];
			dst[offset + 3 * stride] = load[1] - load[2] + c2 * (load[3] - load[4] + load[5]);
		}
		__device__ T& operator[](unsigned int idx) noexcept
		{
			assert(idx < 6);
			return load[idx];
		}
	};

#define outputTensorIndex(b, h, w, f) ((((b) * shape.y + (h)) * shape.z + (w)) * shape.w + (f))

	template<unsigned int winogradTileSize, unsigned int kernelSize, unsigned int elements, typename T, typename U = T>
	__global__ void kernel_winograd_output_transform(const T* matrices, T* output, uint4 shape, const T* bias, avActivationType_t activation)
	{
		assert(blockDim.x == elements && blockDim.y == 1);
		constexpr unsigned int tile_size = winogradTileSize + kernelSize - 1;
		__shared__ U storage[winogradTileSize * tile_size * elements];

//		for (unsigned int f = threadIdx.x; f < shape.w; f += elements)
		for (unsigned int f = threadIdx.x; f < 1; f += elements)
		{
			OutputTransform<winogradTileSize, kernelSize, U> tile;
			for (int w = 0; w < tile_size; w++)
			{
				unsigned int stride = gridDim.x * gridDim.y * gridDim.z * shape.w;
				unsigned int offset = tile_size * w * stride + ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.w + f;

				for (int h = 0; h < tile_size; h++)
					tile[h] = matrices[offset + h * stride];

				for (int h = 0; h < tile_size; h++)
					printf("%f ", tile[h]);
				printf("\n");

//				printf("%i %i\n", offset, stride);
				offset = w * elements + threadIdx.x;
				stride = tile_size * elements;
				tile.transformInto(storage, offset, stride);
			}

			printf("---------------------------------\n");
			for (int w = 0; w < winogradTileSize; w++)
			{
				for (int h = 0; h < tile_size; h++)
					printf("%f ", storage[(w * tile_size + h) * elements]);
				printf("\n");
			}
			printf("---------------------------------\n");

			for (int w = 0; w < winogradTileSize; w++)
			{
				for (int h = 0; h < winogradTileSize; h++)
				{
					int b = blockIdx.x;
					int x = winogradTileSize * blockIdx.y + h;
					int y = winogradTileSize * blockIdx.z + w;
//					printf("%i %i %i %i\n", b, x, y, f);
//					if (x >= 0 and x < shape.y and y >= 0 and y < shape.z)
//						output[inputTensorIndex(b, x, y, f)] = tile[h];
				}

//				for (int h = 0; h < tile_size; h++)
//					printf("%f ", tile[h]);
//				printf("\n");
				unsigned int offset = w * elements + threadIdx.x;
				unsigned int stride = tile_size * elements;
				tile.transformInto(storage, offset, stride);
			}
//

//
//			for (int w = 0; w < tile_size; w++)
//			{
//				unsigned int offset = w * tile_size * elements + threadIdx.x;
//				unsigned int stride = elements;
//				for (int h = 0; h < tile_size; h++)
//					tile[h] = storage[offset + h * stride];
//
////				for (int h = 0; h < tile_size; h++)
////					printf("%f ", tile[h]);
////				printf("\n");
//
//				stride = gridDim.x * gridDim.y * gridDim.z * shape.w;
//				offset = tile_size * w * stride + ((blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.w + f;
////				printf("%i %i\n", offset, stride);
//				tile.transformInto(matrices, offset, stride);
//			}
		}
	}
#undef inputTensorIndex

	template<typename T>
	__global__ void kernel_conv3x3_4x4_weight_transform(T* matrices, const T* weights, const int filters, bool invert)
	{
		__shared__ T storage[18 * 32];

		T c23 = static_cast<T>(2.0 / 3.0);
		T c13 = static_cast<T>(1.0 / 3.0);
		T c2 = static_cast<T>(2);
		T c4 = static_cast<T>(4);

		for (int f = 0; f < filters; f += 32)
		{
			T load0 = 0, load1 = 0, load2 = 0;

			if (f + threadIdx.x < filters)
			{
				if (invert == false)
				{
					load0 = weights[(blockIdx.x * 9 + (threadIdx.y + 0 * 3)) * filters + f + threadIdx.x];
					load1 = weights[(blockIdx.x * 9 + (threadIdx.y + 1 * 3)) * filters + f + threadIdx.x];
					load2 = weights[(blockIdx.x * 9 + (threadIdx.y + 2 * 3)) * filters + f + threadIdx.x];
				}
				else
				{
					load0 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 0 * 3)) * filters + f + threadIdx.x];
					load1 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 1 * 3)) * filters + f + threadIdx.x];
					load2 = weights[(blockIdx.x * 9 + 8 - (threadIdx.y + 2 * 3)) * filters + f + threadIdx.x];
				}
			}

			int tmp_idx = threadIdx.y * 32 + threadIdx.x;
			storage[tmp_idx] = load0;
			storage[tmp_idx + 96] = c23 * (load0 + load1 + load2);
			storage[tmp_idx + 192] = c23 * (load0 - load1 + load2);
			storage[tmp_idx + 288] = c13 * (load0 + c2 * load1 + c4 * load2);
			storage[tmp_idx + 384] = c13 * (load0 - c2 * load1 + c4 * load2);
			storage[tmp_idx + 480] = c2 * load2;
			__syncthreads();

			for (int k = threadIdx.y; k < 6; k += 3)
			{
				tmp_idx = k * 96 + threadIdx.x;
				load0 = storage[tmp_idx];
				load1 = storage[tmp_idx + 32];
				load2 = storage[tmp_idx + 64];

				tmp_idx = ((6 * k + 0) * gridDim.x + blockIdx.x) * filters + f + threadIdx.x;
				if (f + threadIdx.x < filters)
				{
					matrices[tmp_idx + 0 * gridDim.x * filters] = load0;
					matrices[tmp_idx + 1 * gridDim.x * filters] = c23 * (load0 + load1 + load2);
					matrices[tmp_idx + 2 * gridDim.x * filters] = c23 * (load0 - load1 + load2);
					matrices[tmp_idx + 3 * gridDim.x * filters] = c13 * (load0 + c2 * load1 + c4 * load2);
					matrices[tmp_idx + 4 * gridDim.x * filters] = c13 * (load0 - c2 * load1 + c4 * load2);
					matrices[tmp_idx + 5 * gridDim.x * filters] = c2 * load2;
				}
			}
			__syncthreads();
		}
	}
	template<int tile_length>
	__launch_bounds__(384, 5)
	__global__ void kernel_conv3x3_4x4_input_transform(float* matrices, const float* input, int3 shape)
	{
		__shared__ float data[36][tile_length];

		for (int f = 0; f < shape.z; f += tile_length)
		{
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int h = 4 * blockIdx.y - 1 + i / 6;
				int w = 4 * blockIdx.z - 1 + i % 6;
				if (h >= 0 && h < shape.x && w >= 0 && w < shape.y)
				{
					int filter_id = f + threadIdx.x;
					int tmp_idx = ((blockIdx.x * shape.x + h) * shape.y + w) * shape.z + filter_id;
					if (filter_id < shape.z)
						data[i][threadIdx.x] = input[tmp_idx];
					if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
						data[i][threadIdx.x + blockDim.x] = input[tmp_idx + blockDim.x];
				}
				else
				{
					data[i][threadIdx.x] = 0.0f;
					if (threadIdx.x + blockDim.x < tile_length)
						data[i][threadIdx.x + blockDim.x] = 0.0f;
				}
			}
			__syncthreads();
			for (int i = threadIdx.x; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = 6 * threadIdx.y;
				float load0 = data[tmp_idx + 0][i];
				float load1 = data[tmp_idx + 1][i];
				float load2 = data[tmp_idx + 2][i];
				float load3 = data[tmp_idx + 3][i];
				float load4 = data[tmp_idx + 4][i];
				float load5 = data[tmp_idx + 5][i];
				__syncthreads();

				data[tmp_idx + 0][i] = load0 - load2 + 0.25f * (load4 - load2);
				data[tmp_idx + 1][i] = load1 + load2 - 0.25f * (load3 + load4);
				data[tmp_idx + 2][i] = load2 - load1 + 0.25f * (load3 - load4);
				data[tmp_idx + 3][i] = load3 - load1 + 0.5f * (load4 - load2);
				data[tmp_idx + 4][i] = load1 - load3 + 0.5f * (load4 - load2);
				data[tmp_idx + 5][i] = load1 - load3 + 0.25f * (load5 - load3);
			}
			__syncthreads();
			for (int i = threadIdx.x; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = threadIdx.y;
				float load0 = data[tmp_idx + 0][i];
				float load1 = data[tmp_idx + 6][i];
				float load2 = data[tmp_idx + 12][i];
				float load3 = data[tmp_idx + 18][i];
				float load4 = data[tmp_idx + 24][i];
				float load5 = data[tmp_idx + 30][i];
				__syncthreads();

				data[tmp_idx + 0][i] = load0 - load2 + 0.25f * (load4 - load2);
				data[tmp_idx + 6][i] = load1 + load2 - 0.25f * (load3 + load4);
				data[tmp_idx + 12][i] = load2 - load1 + 0.25f * (load3 - load4);
				data[tmp_idx + 18][i] = load3 - load1 + 0.5f * (load4 - load2);
				data[tmp_idx + 24][i] = load1 - load3 + 0.5f * (load4 - load2);
				data[tmp_idx + 30][i] = load1 - load3 + 0.25f * (load5 - load3);
			}
			__syncthreads();
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int filter_id = f + threadIdx.x;
				int tmp_idx = (((blockIdx.x + i * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.z + filter_id;
				if (filter_id < shape.z)
					matrices[tmp_idx] = data[i][threadIdx.x];
				if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
					matrices[tmp_idx + blockDim.x] = data[i][threadIdx.x + blockDim.x];
			}
			__syncthreads();
		}
	}
	template<int tile_length>
	__launch_bounds__(384, 5)
	__global__ void kernel_conv3x3_4x4_output_transform(const float* matrices, float* output, int3 shape, const float *biases, const float *add,
			avActivationType_t act)
	{
		__shared__ float data[36][tile_length];

		for (int f = 0; f < shape.z; f += tile_length)
		{
			for (int i = threadIdx.y; i < 36; i += 6)
			{
				int filter_id = f + threadIdx.x;
				int tmp_idx = (((blockIdx.x + i * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * shape.z + filter_id;
				if (filter_id < shape.z)
					data[i][threadIdx.x] = matrices[tmp_idx];
				if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
					data[i][threadIdx.x + blockDim.x] = matrices[tmp_idx + tile_length / 2];
			}
			__syncthreads();
			for (int i = 0; i < tile_length; i += blockDim.x)
			{
				int tmp_idx = 6 * threadIdx.y;
				float load0 = data[tmp_idx + 0][i + threadIdx.x];
				float load1 = data[tmp_idx + 1][i + threadIdx.x];
				float load2 = data[tmp_idx + 2][i + threadIdx.x];
				float load3 = data[tmp_idx + 3][i + threadIdx.x];
				float load4 = data[tmp_idx + 4][i + threadIdx.x];
				float load5 = data[tmp_idx + 5][i + threadIdx.x];
				__syncthreads();

				data[tmp_idx + 0][i + threadIdx.x] = load0 + load1 + load2 + 0.25f * (load3 + load4);
				data[tmp_idx + 1][i + threadIdx.x] = load1 - load2 + 0.5f * (load3 - load4);
				data[tmp_idx + 2][i + threadIdx.x] = load1 + load2 + load3 + load4;
				data[tmp_idx + 3][i + threadIdx.x] = load1 - load2 + 2.0f * (load3 - load4 + load5);
			}
			__syncthreads();
			for (int i = 0; i < tile_length; i += blockDim.x)
			{
				float bias = 0.0f;
				if (biases != nullptr && (f + i + threadIdx.x) < shape.z)
					bias = biases[f + i + threadIdx.x];

				float load0, load1, load2, load3, load4, load5;
				if (threadIdx.y < 4)
				{
					load0 = data[threadIdx.y + 0][i + threadIdx.x];
					load1 = data[threadIdx.y + 6][i + threadIdx.x];
					load2 = data[threadIdx.y + 12][i + threadIdx.x];
					load3 = data[threadIdx.y + 18][i + threadIdx.x];
					load4 = data[threadIdx.y + 24][i + threadIdx.x];
					load5 = data[threadIdx.y + 30][i + threadIdx.x];
				}
				__syncthreads();
				if (threadIdx.y < 4)
				{
					data[threadIdx.y + 0][i + threadIdx.x] = bias + load0 + load1 + load2 + 0.25f * (load3 + load4);
					data[threadIdx.y + 4][i + threadIdx.x] = bias + load1 - load2 + 0.5f * (load3 - load4);
					data[threadIdx.y + 8][i + threadIdx.x] = bias + load1 + load2 + load3 + load4;
					data[threadIdx.y + 12][i + threadIdx.x] = bias + load1 - load2 + 2.0f * (load3 - load4 + load5);
				}
			}
			__syncthreads();
			if (threadIdx.y < 4)
				for (int i = threadIdx.y; i < 16; i += 4)
				{
					int h = 4 * blockIdx.y + i / 4;
					int w = 4 * blockIdx.z + i % 4;
					if (h < shape.x && w < shape.y)
					{
						int filter_id = f + threadIdx.x;
						int tmp_idx = ((blockIdx.x * shape.x + h) * shape.y + w) * shape.z + filter_id;

						if (filter_id < shape.z)
						{
							if (add != nullptr)
								data[i][threadIdx.x] += add[tmp_idx];
							output[tmp_idx] = device_act_forward(act, data[i][threadIdx.x]);
						}
						if (filter_id + blockDim.x < shape.z && threadIdx.x + blockDim.x < tile_length)
						{
							if (add != nullptr)
								data[i][threadIdx.x + blockDim.x] += add[tmp_idx + blockDim.x];
							output[tmp_idx + blockDim.x] = device_act_forward(act, data[i][threadIdx.x + blockDim.x]);
						}
					}
				}
			__syncthreads();
		}
	}
	__global__ void kernel_conv3x3_4x4_gradient_transform(float *matrices, const float *gradient, int height, int width, int filters)
	{
		__shared__ int indices_in[16];
		__shared__ int indices_out[36];
		__shared__ float data[36 * 32];
		int tid = 32 * threadIdx.y + threadIdx.x; //192 threads

		if (tid < 16)
		{
			int h = 4 * blockIdx.y + tid / 4;
			int w = 4 * blockIdx.z + tid % 4;
			if (h < height && w < width)
				indices_in[tid] = tensorIdx4D(blockIdx.x, h, w, 0);
			else
				indices_in[tid] = -1;
		}
		if (tid < 36)
		{
			indices_out[tid] = (((blockIdx.x + tid * gridDim.x) * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z) * filters;
		}
		for (int f = threadIdx.x; f < filters; f += 32)
		{
			__syncthreads();
			for (int i = threadIdx.y; i < 16; i += 6)
				if (indices_in[i] != -1)
					data[32 * i + threadIdx.x] = gradient[indices_in[i] + f];
				else
					data[32 * i + threadIdx.x] = 0.0f;
			__syncthreads();
			float load0, load1, load2, load3;
			if (threadIdx.y < 4)
			{
				int tmp_idx = 32 * threadIdx.y + threadIdx.x;
				load0 = data[tmp_idx + 0];
				load1 = data[tmp_idx + 128];
				load2 = data[tmp_idx + 256];
				load3 = data[tmp_idx + 384];
			}
			__syncthreads();

			if (threadIdx.y < 4)
			{
				int tmp_idx = 32 * threadIdx.y + threadIdx.x;
				float tmp0 = 2.0f * load3; //2*load3
				float tmp1 = load0 + load2; //load0+load2
				float tmp2 = load1 + load3; //load1+load3
				float tmp3 = 0.333333f * tmp1;
				tmp1 = 0.666667f * tmp1;
				tmp2 = 0.666667f * tmp2;
				tmp3 = tmp3 + load2;
				float tmp4 = tmp2 + tmp0;

				data[tmp_idx] = load0;
				data[tmp_idx + 128] = tmp1 + tmp2;
				data[tmp_idx + 256] = tmp1 - tmp2;
				data[tmp_idx + 384] = tmp3 + tmp4;
				data[tmp_idx + 512] = tmp3 - tmp4;
				data[tmp_idx + 640] = tmp0;
			}
			__syncthreads();

			int tmp_idx = 128 * threadIdx.y + threadIdx.x;
			load0 = data[tmp_idx];
			load1 = data[tmp_idx + 32];
			load2 = data[tmp_idx + 64];
			load3 = data[tmp_idx + 96];

			float tmp0 = 2.0f * load3; //2*load3
			float tmp1 = load0 + load2; //load0+load2
			float tmp2 = load1 + load3; //load1+load3
			float tmp3 = 0.333333f * tmp1;
			tmp1 = 0.666667f * tmp1;
			tmp2 = 0.666667f * tmp2;
			tmp3 = tmp3 + load2;
			float tmp4 = tmp2 + tmp0;

			matrices[indices_out[6 * threadIdx.y + 0] + f] = load0;
			matrices[indices_out[6 * threadIdx.y + 1] + f] = tmp1 + tmp2;
			matrices[indices_out[6 * threadIdx.y + 2] + f] = tmp1 - tmp2;
			matrices[indices_out[6 * threadIdx.y + 3] + f] = tmp3 + tmp4;
			matrices[indices_out[6 * threadIdx.y + 4] + f] = tmp3 - tmp4;
			matrices[indices_out[6 * threadIdx.y + 5] + f] = tmp0;
		}
	}
	__global__ void kernel_conv3x3_4x4_update_transform(const float *matrices, float *update, int filters)
	{
		__shared__ int indices_in[36];
		__shared__ float data[36 * 32];
		int tid = 32 * threadIdx.y + threadIdx.x; //192 threads
		if (tid < 36)
			indices_in[tid] = (tid * gridDim.x + blockIdx.x) * filters;

		for (int f = threadIdx.x; f < filters; f += 32)
		{
			__syncthreads();
			for (int i = threadIdx.y; i < 36; i += 6)
				data[32 * i + threadIdx.x] = matrices[indices_in[i] + f];
			__syncthreads();

			int tmp_idx = 32 * threadIdx.y + threadIdx.x;
			float load0 = data[tmp_idx];
			float load1 = data[tmp_idx + 192];
			float load2 = data[tmp_idx + 384];
			float load3 = data[tmp_idx + 576];
			float load4 = data[tmp_idx + 768];
			float load5 = data[tmp_idx + 960];

			float tmp1 = load1 + load2;
			float tmp2 = load1 - load2;
			float tmp3 = load3 + load4;
			float tmp4 = load3 - load4;
			load0 += tmp1 + 0.25f * tmp3;
			load1 = tmp2 + 0.5f * tmp4;
			load2 = tmp1 + tmp3 + 2.0f * load5;

			__syncthreads();
			data[tmp_idx] = load0;
			data[tmp_idx + 192] = load1;
			data[tmp_idx + 384] = load2;
			__syncthreads();

			if (threadIdx.y < 3)
			{
				tmp_idx = 192 * threadIdx.y + threadIdx.x;
				load0 = data[tmp_idx];
				load1 = data[tmp_idx + 32];
				load2 = data[tmp_idx + 64];
				load3 = data[tmp_idx + 96];
				load4 = data[tmp_idx + 128];
				load5 = data[tmp_idx + 160];

				tmp1 = load1 + load2;
				tmp2 = load1 - load2;
				tmp3 = load3 + load4;
				tmp4 = load3 - load4;

				load0 += tmp1 + 0.25f * tmp3;
				load1 = tmp2 + 0.5f * tmp4;
				load2 = tmp1 + tmp3 + 2.0f * load5;

				update[(blockIdx.x * 9 + 3 * threadIdx.y + 0) * filters + f] += load0;
				update[(blockIdx.x * 9 + 3 * threadIdx.y + 1) * filters + f] += load1;
				update[(blockIdx.x * 9 + 3 * threadIdx.y + 2) * filters + f] += load2;
			}
		}
	}
}

namespace avocado
{
	namespace backend
	{

		avStatus_t cudaGetConvolutionWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t bDesc, avSize_t *result)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaPrecomputeConvolutionWorkspace(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, avMemoryDescriptor_t workspaceMem)
		{
			int filters_out = getTensor(wDesc).firstDim();
			int filters_in = getTensor(wDesc).lastDim();
//			dim3 blockSize(32, 3);
			dim3 blockDim(128);
			dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
			cudaStream_t stream = getContext(context).getStream();

			switch (getTensor(wDesc).dtype())
			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
//							reinterpret_cast<const half*>(weight->data), filters_in, invert);
//					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_winograd_weight_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(workspaceMem), getPointer<float>(wMem),
							filters_out, filters_in, false);
					break;
				case AVOCADO_DTYPE_FLOAT64:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
//							reinterpret_cast<const double*>(weight->data), filters_in, invert);
					break;
			}
//			return cudaGetLastError();
			cudaStreamSynchronize(stream);
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation, avMemoryDescriptor_t workspaceMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, avMemoryDescriptor_t workspaceMem)
		{
			uint4 shape { getTensor(yDesc).dimension(0), getTensor(yDesc).dimension(1), getTensor(yDesc).dimension(2), getTensor(yDesc).dimension(3) };
			dim3 blockDim(1);
			dim3 gridDim(shape.x, (shape.y + 3) / 4, (shape.z + 3) / 4);
			cudaStream_t stream = getContext(context).getStream();

			switch (getTensor(xDesc).dtype())
			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
//							reinterpret_cast<const half*>(weight->data), filters_in, invert);
//					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_winograd_output_transform<4, 3, 128, float> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(workspaceMem),
							getPointer<float>(yMem), shape, nullptr, AVOCADO_ACTIVATION_LINEAR);
					break;
				case AVOCADO_DTYPE_FLOAT64:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
//							reinterpret_cast<const double*>(weight->data), filters_in, invert);
					break;
			}
			cudaStreamSynchronize(stream);

//			uint4 shape { getTensor(xDesc).dimension(0), getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//			int2 padding { -1, -1 };
//			dim3 blockDim(128);
//			dim3 gridDim(shape.x, (shape.y + 3) / 4, (shape.z + 3) / 4);
//			cudaStream_t stream = getContext(context).getStream();
//
//			switch (getTensor(xDesc).dtype())
//			{
////				case AVOCADO_DTYPE_FLOAT16:
////					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
////							reinterpret_cast<const half*>(weight->data), filters_in, invert);
////					break;
//				case AVOCADO_DTYPE_FLOAT32:
//					kernel_winograd_input_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(workspaceMem), getPointer<float>(xMem),
//							shape, padding, 0.0f);
//					break;
//				case AVOCADO_DTYPE_FLOAT64:
////					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<double*>(matrices->data),
////							reinterpret_cast<const double*>(weight->data), filters_in, invert);
//					break;
//			}
//			cudaStreamSynchronize(stream);

//			int3 shape { getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//
//			int tiles_h = (getTensor(xDesc).dimension(1) + 3) / 4;
//			int tiles_w = (getTensor(xDesc).dimension(2) + 3) / 4;
//			dim3 gridSize(getTensor(xDesc).firstDim(), tiles_h, tiles_w);
//			cudaStream_t stream = getContext(context).getStream();
//
//			dim3 blockSize(32, 6);
//			kernel_conv3x3_4x4_input_transform<64> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(workspaceMem), getPointer<float>(xMem), shape);
//			cudaStreamSynchronize(stream);
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaConvolutionUpdate(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem,
				const void *beta, const avTensorDescriptor_t dwDesc, avMemoryDescriptor_t dwMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}
	} /* namespace backend */
} /* namespace avocado */

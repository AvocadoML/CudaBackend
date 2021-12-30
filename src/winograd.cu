/*
 * winograd.cu
 *
 *  Created on: Dec 29, 2021
 *      Author: Maciej Kozarzewski
 */

#include <avocado/backend/backend_descriptors.hpp>

#include "winograd.hpp"
#include "activations.cuh"
#include "utilities.hpp"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>

namespace
{
	using namespace avocado::backend;

	enum class TransformType
	{
		WEIGHT, INPUT, OUTPUT, GRADIENT, UPDATE
	};

#define tensorIndex(b, h, w, f) ((((b) * shape.y + (h)) * shape.z + (w)) * shape.w + (f))
#define matrixIndex(i, j, k) (((i) * matrix_shape.y + (j)) * matrix_shape.z + (k))

	struct TensorShape
	{
		int batch, height, width, filters;

		__host__ __device__ int offset_at(int b, int h, int w, int f) const noexcept
		{
			return ((b * height + h) * width + w) * filters + f;
		}
		__host__ __device__ int stride_to_next_row() const noexcept
		{
			return width * filters;
		}
		__host__ __device__ int stride_to_next_column() const noexcept
		{
			return filters;
		}
	};
	template<int TileSize>
	struct MatrixShape
	{
		int tiles, filters;

		__host__ __device__ int offset_at(int r, int c, int t, int f) const noexcept
		{
			return ((r * TileSize + c) * tiles + t) * filters + f;
		}
		__host__ __device__ int stride_to_next_row() const noexcept
		{
			return TileSize * tiles * filters;
		}
		__host__ __device__ int stride_to_next_column() const noexcept
		{
			return tiles * filters;
		}
	};
	template<int Rows, int Columns, int Elements>
	struct SharedArrayShape
	{
		__host__ __device__ int offset_at(int r, int c, int e) const noexcept
		{
			return (r * Columns + c) * Elements + e;
		}
		__host__ __device__ int stride_to_next_row() const noexcept
		{
			return Columns * Elements;
		}
		__host__ __device__ int stride_to_next_column() const noexcept
		{
			return Elements;
		}
	};

	template<int Size, typename T>
	struct Tile
	{
	private:
		T data[Size];
	public:
//		template<int ElementsToLoad>
//		__host__ __device__ void load(const T* src, int offset, int stride) noexcept
//		{
//			for (int i = 0; i < ElementsToLoad; i++)
//				data[i] = src[offset + i * stride];
//		}
//		template<int ElementsToStore>
//		__host__ __device__ void store(T* src, int offset, int stride) const noexcept
//		{
//			for (int i = 0; i < ElementsToStore; i++)
//				src[offset + i * stride] = data[i];
//		}
		__host__ __device__ void load(const T* src, int offset, int stride) noexcept
		{
			for (int i = 0; i < Size; i++)
				data[i] = src[offset + i * stride];
		}
		__host__ __device__ void store(T* src, int offset, int stride) const noexcept
		{
			for (int i = 0; i < Size; i++)
				src[offset + i * stride] = data[i];
		}
		__host__ __device__ T& operator[](int index) noexcept
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		__host__ __device__ T operator[](int index) const noexcept
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		Tile<Size, T>& operator+=(T x) noexcept
		{
			for (int i = 0; i < Size; i++)
				data[i] += x;
			return *this;
		}
	};

	// 1.0  0.0  0.0
	// 2/3  2/3  2/3
	// 2/3 -2/3  2/3
	// 1/3  2/3  4/3
	// 1/3 -2/3  4/3
	// 0.0  0.0  2.0

	// 1.0  0.0 -1.25  0.0   0.25  0.0
	// 0.0  1.0  1.0  -0.25 -0.25  0.0
	// 0.0 -1.0  1.0   0.25 -0.25  0.0
	// 0.0 -1.0 -0.5   1.0   0.5   0.0
	// 0.0  1.0 -0.5  -1.0   0.5   0.0
	// 0.0  1.0  0.0  -1.25  0.0   0.25

	// 1.0 1.0  1.0 0.25 0.25 0.0
	// 0.0 1.0 -1.0 0.5 -0.5  0.0
	// 0.0 1.0  1.0 1.0  1.0  0.0
	// 0.0 1.0 -1.0 2.0 -2.0  2.0

	// 1.0  0.0  0.0  0.0
	// 2/3  2/3  2/3  2/3
	// 2/3 -2/3  2/3 -2/3
	// 1/3  2/3  4/3  8/3
	// 1/3 -2/3  4/3 -8/3
	// 0.0  0.0  0.0  2.0

	// 1.0  1.0  1.0  0.25 0.25 0.0
	// 0.0  1.0 -1.0  0.5 -0.5  0.0
	// 0.0  1.0  1.0  1.0  1.0  2.0

	template<TransformType Type, int TransformSize, int KernelSize, typename T>
	struct TileTransform
	{
	};
	template<typename T>
	struct TileTransform<TransformType::WEIGHT, 4, 3, T>
	{
		__host__ __device__ Tile<6, T> operator()(const Tile<3, T> &tile) const noexcept
		{
			T c13 = static_cast<T>(1.0 / 3.0);
			T c23 = static_cast<T>(2.0 / 3.0);
			T c2 = static_cast<T>(2.0);
			T c4 = static_cast<T>(4.0);

			T load0 = tile[0];
			T load1 = tile[1];
			T load2 = tile[2];

			Tile<6, T> result;
			result[0] = load0;
			result[1] = c23 * (load0 + load1 + load2);
			result[2] = c23 * (load0 - load1 + load2);
			result[3] = c13 * (load0 + c2 * load1 + c4 * load2);
			result[4] = c13 * (load0 - c2 * load1 + c4 * load2);
			result[5] = c2 * load2;
			return result;
		}
		__host__ __device__ T operator()(const Tile<3, T> &tile, int row) const noexcept
		{
			// 1.0  0.0  0.0
			// 2/3  2/3  2/3
			// 2/3 -2/3  2/3
			// 1/3  2/3  4/3
			// 1/3 -2/3  4/3
			// 0.0  0.0  2.0

			switch (row)
			{
				case 0:
					return tile[0];
				case 1:
					return static_cast<T>(2.0 / 3.0) * (tile[0] + tile[1] + tile[2]);
				case 2:
					return static_cast<T>(2.0 / 3.0) * (tile[0] - tile[1] + tile[2]);
				case 3:
					return static_cast<T>(1.0 / 3.0) * (tile[0] + static_cast<T>(2.0) * tile[1] + static_cast<T>(4.0) * tile[2]);
				case 4:
					return static_cast<T>(1.0 / 3.0) * (tile[0] - static_cast<T>(2.0) * tile[1] + static_cast<T>(4.0) * tile[2]);
				case 5:
					return static_cast<T>(2.0) * tile[2];
			}
			return zero<T>();
		}
//		__host__ __device__ void operator()(Tile<6, T> &tile) const noexcept
//		{
//			T c13 = static_cast<T>(1.0 / 3.0);
//			T c23 = static_cast<T>(2.0 / 3.0);
//			T c2 = static_cast<T>(2.0);
//			T c4 = static_cast<T>(4.0);
//
//			T load0 = tile[0];
//			T load1 = tile[1];
//			T load2 = tile[2];
//
//			tile[0] = load0;
//			tile[1] = c23 * (load0 + load1 + load2);
//			tile[2] = c23 * (load0 - load1 + load2);
//			tile[3] = c13 * (load0 + c2 * load1 + c4 * load2);
//			tile[4] = c13 * (load0 - c2 * load1 + c4 * load2);
//			tile[5] = c2 * load2;
//		}
	};
	template<typename T>
	struct TileTransform<TransformType::INPUT, 4, 3, T>
	{
		__host__ __device__ void operator()(Tile<6, T> &tile) const noexcept
		{
			T c025 = static_cast<T>(0.25);
			T c05 = static_cast<T>(0.5);

			T load0 = tile[0];
			T load1 = tile[1];
			T load2 = tile[2];
			T load3 = tile[3];
			T load4 = tile[4];
			T load5 = tile[5];

			tile[0] = load0 - load2 + c025 * (load4 - load2);
			tile[1] = load1 + load2 - c025 * (load3 + load4);
			tile[2] = load2 - load1 + c025 * (load3 - load4);
			tile[3] = load3 - load1 + c05 * (load4 - load2);
			tile[4] = load1 - load3 + c05 * (load4 - load2);
			tile[5] = load1 - load3 + c025 * (load5 - load3);
		}
	};
	template<typename T>
	struct TileTransform<TransformType::OUTPUT, 4, 3, T>
	{
		__host__ __device__ void operator()(Tile<6, T> &tile) const noexcept
		{
			T c025 = static_cast<T>(0.25);
			T c05 = static_cast<T>(0.5);
			T c2 = static_cast<T>(2.0);

			T load0 = tile[0];
			T load1 = tile[1];
			T load2 = tile[2];
			T load3 = tile[3];
			T load4 = tile[4];
			T load5 = tile[5];

			tile[0] = load0 + load1 + load2 + c025 * (load3 + load4);
			tile[1] = load1 - load2 + c05 * (load3 - load4);
			tile[2] = load1 + load2 + load3 + load4;
			tile[3] = load1 - load2 + c2 * (load3 - load4 + load5);
		}
	};
	template<typename T>
	struct TileTransform<TransformType::GRADIENT, 4, 3, T>
	{
		__host__ __device__ void operator()(Tile<6, T> &tile) const noexcept
		{
			T c13 = static_cast<T>(1.0 / 3.0);
			T c23 = static_cast<T>(2.0 / 3.0);
			T c2 = static_cast<T>(2.0);
			T c4 = static_cast<T>(4.0);
			T c8 = static_cast<T>(8.0);

			T load0 = tile[0];
			T load1 = tile[1];
			T load2 = tile[2];
			T load3 = tile[3];

			tile[0] = load0;
			tile[1] = c23 * (load0 + load1 + load2 + load3);
			tile[2] = c23 * (load0 - load1 + load2 - load3);
			tile[3] = c13 * (load0 + c2 * load1 + c4 * load2 + c8 * load3);
			tile[4] = c13 * (load0 - c2 * load1 + c4 * load2 - c8 * load3);
			tile[5] = c2 * load3;
		}
	};
	template<typename T>
	struct TileTransform<TransformType::UPDATE, 4, 3, T>
	{
		__host__ __device__ void operator()(Tile<6, T> &tile) const noexcept
		{
			T c025 = static_cast<T>(0.25);
			T c05 = static_cast<T>(0.5);
			T c2 = static_cast<T>(2.0);

			// 1.0  1.0  1.0  0.25 0.25 0.0
			// 0.0  1.0 -1.0  0.5 -0.5  0.0
			// 0.0  1.0  1.0  1.0  1.0  2.0

			T load0 = tile[0];
			T load1 = tile[1];
			T load2 = tile[2];
			T load3 = tile[3];
			T load4 = tile[4];
			T load5 = tile[5];

			tile[0] = load0 + load1 + load2 + c025 * (load3 + load4);
			tile[1] = load1 - load2 + c05 * (-load3 + load4);
			tile[2] = load1 + load2 + load3 + load4 + c2 * load5;
		}
	};

	template<int TransformSize, int KernelSize, unsigned int Elements, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_weight_transform2(T* matrices, MatrixShape<TileSize> matrix_shape, const T* weights, TensorShape tensor_shape, bool invert)
	{
		assert(blockDim.x == Elements && blockDim.y == TransformSize);
		__shared__ T storage[KernelSize * KernelSize * Elements];

		for (unsigned int filter = blockIdx.x * blockDim.x + threadIdx.x; filter < tensor_shape.filters; filter += gridDim.x * blockDim.x)
		{
			for (unsigned int i = threadIdx.y; i < square(KernelSize); i += blockDim.y)
			{
				SharedArrayShape<KernelSize, KernelSize, Elements> shared_indexer;
				int idx = invert ? (square(KernelSize) - 1 - i) : i;
				int row = idx / KernelSize;
				int col = idx % KernelSize;
				int tensor_offset = tensor_shape.offset_at(blockIdx.y, row, col, filter);
				int storage_offset = shared_indexer.offset_at(row, col, threadIdx.x);
				storage[storage_offset] = weights[tensor_offset];
			}
			__syncthreads();

			Tile<KernelSize, T> computed_row;
			for (unsigned int col = 0; col < KernelSize; col++)
			{
				SharedArrayShape<KernelSize, KernelSize, Elements> shared_indexer;
				int offset = shared_indexer.offset_at(0, col, threadIdx.x);
				int stride = shared_indexer.stride_to_next_row();

				Tile<KernelSize, T> loaded_column;
				loaded_column.load(storage, offset, stride);

				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
				computed_row[col] = transform(loaded_column, threadIdx.y);
			}

			for (unsigned int col = 0; col < TileSize; col++)
			{
				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
				T tmp = transform(computed_row, col);

				int tile_index = blockIdx.y;
				int offset = matrix_shape.offset_at(threadIdx.y, col, tile_index, filter);
				matrices[offset] = tmp;
			}
			__syncthreads();
		}
	}

	template<int TransformSize, int KernelSize, unsigned int Elements, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_weight_transform(T* matrices, MatrixShape<TileSize> matrix_shape, const T* weights, TensorShape tensor_shape, bool invert)
	{
		assert(blockDim.x == Elements && blockDim.y == 1);
		__shared__ T storage[TileSize * KernelSize * Elements];

		for (unsigned int filter = blockIdx.x * blockDim.x + threadIdx.x; filter < tensor_shape.filters; filter += gridDim.x * blockDim.x)
		{
			for (unsigned int column = 0; column < KernelSize; column++)
			{
				int offset, stride;
				if (invert)
				{
					offset = tensor_shape.offset_at(blockIdx.y, KernelSize - 1, KernelSize - 1 - column, filter);
					stride = -tensor_shape.stride_to_next_row();
				}
				else
				{
					offset = tensor_shape.offset_at(blockIdx.y, 0, column, filter);
					stride = tensor_shape.stride_to_next_row();
				}
				Tile<KernelSize, T> tile;
				tile.load(weights, offset, stride);

				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
				Tile<TileSize, T> transformed = transform(tile);

				SharedArrayShape<TileSize, KernelSize, Elements> shared_indexer;
				offset = shared_indexer.offset_at(0, column, threadIdx.x);
				stride = shared_indexer.stride_to_next_row();
				transformed.store(storage, offset, stride);

//				Tile<TileSize, T> tile;
//				tile.load<KernelSize>(weights, offset, stride);
//
//				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
//				transform(tile);
//
//				SharedArrayShape<TileSize, KernelSize, Elements> shared_indexer;
//				offset = shared_indexer.offset_at(0, column, threadIdx.x);
//				stride = shared_indexer.stride_to_next_row();
//				tile.store<TileSize>(storage, offset, stride);
			}

			for (unsigned int row = 0; row < TileSize; row++)
			{
				SharedArrayShape<TileSize, KernelSize, Elements> shared_indexer;
				int offset = shared_indexer.offset_at(row, 0, threadIdx.x);
				int stride = shared_indexer.stride_to_next_column();

				Tile<KernelSize, T> tile;
				tile.load(storage, offset, stride);

				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
				Tile<TileSize, T> transformed = transform(tile);

				int tile_index = blockIdx.y;
				offset = matrix_shape.offset_at(row, 0, tile_index, filter);
				stride = matrix_shape.stride_to_next_column();
				transformed.store(matrices, offset, stride);

//				Tile<TileSize, T> tile;
//				tile.load<KernelSize>(storage, offset, stride);
//
//				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, T> transform;
//				transform(tile);
//
//				int tile_index = blockIdx.y;
//				offset = matrix_shape.offset_at(row, 0, tile_index, filter);
//				stride = matrix_shape.stride_to_next_column();
//				tile.store<TileSize>(matrices, offset, stride);
			}
		}
	}

//	template<typename T>
//	struct WeightTransform2<4, 3, T>
//	{
//	private:
//		T data[6];
//	public:
//		__host__ __device__ void transform() noexcept
//		{
//			T c13 = static_cast<T>(1.0 / 3.0);
//			T c23 = static_cast<T>(2.0 / 3.0);
//			T c2 = static_cast<T>(2.0);
//			T c4 = static_cast<T>(4.0);
//
//			T load0 = data[0];
//			T load1 = data[1];
//			T load2 = data[2];
//
//			data[0] = load0;
//			data[1] = c23 * (load0 + load1 + load2);
//			data[2] = c23 * (load0 - load1 + load2);
//			data[3] = c13 * (load0 + c2 * load1 + c4 * load2);
//			data[4] = c13 * (load0 - c2 * load1 + c4 * load2);
//			data[5] = c2 * load2;
//		}
//		__host__ __device__ T& operator[](unsigned int idx) noexcept
//		{
//			assert(idx < 6);
//			return data[idx];
//		}
//		__host__ __device__ T operator[](unsigned int idx) const noexcept
//		{
//			assert(idx < 6);
//			return data[idx];
//		}
//	};

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

	TensorShape get_tensor_shape(const TensorDescriptor &desc)
	{
		return TensorShape( { desc.dimension(0), desc.dimension(1), desc.dimension(2), desc.dimension(3) });
	}
	template<int TileSize>
	MatrixShape<TileSize> get_matrix_shape(const TensorDescriptor &desc)
	{
		return MatrixShape<TileSize>( { desc.dimension(1), desc.dimension(2) });
	}
}

namespace avocado
{
	namespace backend
	{

		avSize_t winogradGetWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc)
		{
			Tile<6, float> tile;
			tile[0] = 0.0f;
			return 0;
		}

		avStatus_t winogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			TensorShape tensor_shape = get_tensor_shape(getTensor(wDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(getTensor(matricesDesc));

			int filters_out = getTensor(wDesc).firstDim();
			int filters_in = getTensor(wDesc).lastDim();
			dim3 blockDim(64, 6);
			dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
			cudaStream_t stream = getContext(context).getStream();

			switch (getTensor(wDesc).dtype())
			{
//				case AVOCADO_DTYPE_FLOAT16:
//					kernel_conv3x3_4x4_weight_transform<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<half*>(matrices->data),
//							reinterpret_cast<const half*>(weight->data), filters_in, invert);
//					break;
				case AVOCADO_DTYPE_FLOAT32:
					kernel_winograd_weight_transform2<4, 3, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(matricesMem), matrix_shape,
							getPointer<float>(wMem), tensor_shape, false);

//					kernel_winograd_weight_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(matricesMem), matrix_shape,
//							getPointer<float>(wMem), tensor_shape, false);

//					kernel_winograd_weight_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(matricesMem), getPointer<float>(wMem),
//							filters_out, filters_in, false);
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

		avStatus_t winogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t winogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t cudaWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem)
		{
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

	} /* namespace backend */
} /* namespace avocado */

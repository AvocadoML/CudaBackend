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

	__constant__ float weight_transform_4x4_3x3[6 * 3];
	__constant__ float input_transform_4x4_3x3[6 * 6];
	__constant__ float output_transform_4x4_3x3[4 * 6];
	__constant__ float gradient_transform_4x4_3x3[6 * 4];
	__constant__ float update_transform_4x4_3x3[3 * 6];

	enum class TransformType
	{
		WEIGHT, INPUT, OUTPUT, GRADIENT, UPDATE
	};

#define tensorIndex(b, h, w, f) ((((b) * shape.y + (h)) * shape.z + (w)) * shape.w + (f))
#define matrixIndex(i, j, k) (((i) * matrix_shape.y + (j)) * matrix_shape.z + (k))

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
		__device__ int stride_to_next_row() const noexcept
		{
			return width * filters;
		}
		__device__ int stride_to_next_column() const noexcept
		{
			return filters;
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
	template<int TileSize>
	struct MatrixShape
	{
		int tiles, filters;

		__device__ int offset_at(int r, int c, int t, int f) const noexcept
		{
			return ((r * TileSize + c) * tiles + t) * filters + f;
		}
		__device__ int stride_to_next_row() const noexcept
		{
			return TileSize * tiles * filters;
		}
		__device__ int stride_to_next_column() const noexcept
		{
			return tiles * filters;
		}
	};
	template<int Rows, int Columns, int Elements>
	struct SharedArrayShape
	{
		__device__ int offset_at(int r, int c, int e) const noexcept
		{
			return (r * Columns + c) * Elements + e;
		}
		__device__ int stride_to_next_row() const noexcept
		{
			return Columns * Elements;
		}
		__device__ int stride_to_next_column() const noexcept
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
		__device__ void load(const T* src, int offset, int stride) noexcept
		{
			for (int i = 0; i < Size; i++)
				data[i] = src[offset + i * stride];
		}
		__device__ void store(T* src, int offset, int stride) const noexcept
		{
			for (int i = 0; i < Size; i++)
				src[offset + i * stride] = data[i];
		}
		__device__ T& operator[](int index) noexcept
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		__device__ T operator[](int index) const noexcept
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		__device__ Tile<Size, T>& operator+=(T x) noexcept
		{
			for (int i = 0; i < Size; i++)
				data[i] += x;
			return *this;
		}
	};

	template<typename T>
	struct Tensor4DWrapper
	{
	private:
		T* m_data;
	public:
		int batch_size, height, width, filters;

		__device__ int offset_at(int b, int h, int w, int f) const noexcept
		{
			return ((b * height + h) * width + w) * filters + f;
		}
		__device__ int stride_to_next_row() const noexcept
		{
			return width * filters;
		}
		__device__ int stride_to_next_column() const noexcept
		{
			return filters;
		}
	};

	template<int Rows, int Columns, int Elements, typename T>
	struct SharedStorage
	{
		__device__ T* data() noexcept
		{
			extern __shared__ T m_data[];
			return m_data;
		}
		__device__ int offset_at(int row, int col, int element) const noexcept
		{
			assert(row >= 0 && row < Rows);
			assert(col >= 0 && col < Columns);
			assert(element >= 0 && element < Elements);
			return (row * Columns + col) * Elements + element;
		}
		__device__ Tile<Columns, T> get_row(int rowIndex, int elementOffset) noexcept
		{
			Tile<Columns, T> result;
			for (int col = 0; col < Columns; col++)
				result[col] = data()[offset_at(rowIndex, col, elementOffset)];
			return result;
		}
		__device__ Tile<Rows, T> get_column(int columnIndex, int elementOffset) noexcept
		{
			Tile<Columns, T> result;
			for (int row = 0; row < Columns; row++)
				result[row] = data()[offset_at(row, columnIndex, elementOffset)];
			return result;
		}
		__device__ T& at(int row, int col, int element) noexcept
		{
			return data()[offset_at(row, col, element)];
		}
		__device__ T at(int row, int col, int element) const noexcept
		{
			return data()[offset_at(row, col, element)];
		}
		__device__ int stride_to_next_row() const noexcept
		{
			return Columns * Elements;
		}
		__device__ int stride_to_next_column() const noexcept
		{
			return Elements;
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

	/**
	 * \brief Performs the first part of Winograd transform.
	 *
	 * \param[in] src Pointer to memory containing tile data.
	 * \param[in] stride Number of elements between subsequent columns of tile data
	 * \param[in] transformMatrix Pointer to array of transform coefficients.
	 * \param[in] row Selects which row of transform matrix should be multiplied by tile data.
	 */
	template<int Rows, int Columns, typename T>
	__device__ Tile<Columns, T> first_transform(const T* src, const int stride, const T* transformMatrix, const int row) noexcept
	{
		assert(row >= 0 && row < Rows);
		Tile<Columns, T> result;
		for (int i = 0; i < Columns; i++)
			result[i] = zero<T>();
		int index = 0;
		for (int col = 0; col < Columns; col++)
		{
			T c = transformMatrix[row * Columns + col];
			if (c != zero<T>())
				for (int i = 0; i < Columns; i++, index += stride)
					result[i] += c * src[index];
			else
				index += Columns * stride;
		}
		return result;
	}
	/**
	 * \brief Performs the second part of Winograd transform.
	 *
	 * \param[in] src Tile data.
	 * \param[in] transformMatrix Pointer to array of transform coefficients.
	 * \param[in] row Selects which row of transform matrix should be multiplied by tile data.
	 */
	template<int Rows, int Columns, typename T>
	__device__ T second_transform(const Tile<Columns, T> &src, const T* transformMatrix, const int row) noexcept
	{
		assert(row >= 0 && row < Rows);
		T result = zero<T>();
		for (int i = 0; i < Columns; i++)
			result += src[i] * transformMatrix[row * Columns + i];
		return result;
	}

	template<TransformType Type, int TransformSize, int KernelSize, typename T>
	struct TileTransform
	{
	};
	template<typename T>
	struct TileTransform<TransformType::WEIGHT, 4, 3, T>
	{
		__device__ Tile<6, T> operator()(const Tile<3, T> &tile) const noexcept
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
		__device__ T operator()(const Tile<3, T> &tile, int row) const noexcept
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
//		 __device__ void operator()(Tile<6, T> &tile) const noexcept
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
		__device__ Tile<6, T> operator()(const T* src, const int stride, const int row) const noexcept
		{
			return first_transform<6, 6, T>(src, stride, input_transform_4x4_3x3, row);
		}
		__device__ T operator()(const Tile<6, T> &src, const int row) const noexcept
		{
			return second_transform<6, 6, T>(src, input_transform_4x4_3x3, row);
		}
//		__device__ T operator()(const Tile<6, T> &tile, int row) const noexcept
//		{
//			switch (row)
//			{
//				case 0:
//					return tile[0] - tile[2] + static_cast<T>(0.25) * (tile[4] - tile[2]);
//				case 1:
//					return tile[1] + tile[2] - static_cast<T>(0.25) * (tile[3] + tile[4]);
//				case 2:
//					return tile[2] - tile[1] + static_cast<T>(0.25) * (tile[3] - tile[4]);
//				case 3:
//					return tile[3] - tile[1] + static_cast<T>(0.5) * (tile[4] - tile[2]);
//				case 4:
//					return tile[1] - tile[3] + static_cast<T>(0.5) * (tile[4] - tile[2]);
//				case 5:
//					return tile[1] - tile[3] + static_cast<T>(0.25) * (tile[5] - tile[3]);
//			}
//			return zero<T>();
//		}
//		__device__ void operator()(Tile<6, T> &tile) const noexcept
//		{
//			T c025 = static_cast<T>(0.25);
//			T c05 = static_cast<T>(0.5);
//
//			T load0 = tile[0];
//			T load1 = tile[1];
//			T load2 = tile[2];
//			T load3 = tile[3];
//			T load4 = tile[4];
//			T load5 = tile[5];
//
//			tile[0] = load0 - load2 + c025 * (load4 - load2);
//			tile[1] = load1 + load2 - c025 * (load3 + load4);
//			tile[2] = load2 - load1 + c025 * (load3 - load4);
//			tile[3] = load3 - load1 + c05 * (load4 - load2);
//			tile[4] = load1 - load3 + c05 * (load4 - load2);
//			tile[5] = load1 - load3 + c025 * (load5 - load3);
//		}
//		template<int Elements>
//		__device__ void op(Tile<6, T> &result, const T* src, int row) const noexcept
//		{
//			for (int i = 0; i < 6; i++)
//				result[i] = zero<T>();
//			int index = 0;
//			for (int col = 0; col < 6; col++)
//			{
//				T c = static_cast<T>(input_transform_4x4_3x3[row * 6 + col]);
//				if (c != zero<T>())
//					for (int i = 0; i < 6; i++, index += Elements)
//						result[i] += c * src[index];
//				else
//					index += 6 * Elements;
//			}
//		}
//		__device__ void op_transposed(Tile<6, T> &result, const Tile<6, T> &src) const noexcept
//		{
//			for (int i = 0; i < 6; i++)
//				result[i] = zero<T>();
//			int index = 0;
//			for (int col = 0; col < 6; col++)
//			{
//				T tmp = src[col];
//				for (int i = 0; i < 6; i++)
//					result[col] += tmp * static_cast<T>(input_transform_4x4_3x3[6 * i + col]);
//			}
//		}
//		__device__ T operator()(const Tile<6, T> &src, int row) const noexcept
//		{
//			T result = zero<T>();
//			for (int i = 0; i < 6; i++)
//				result += src[i] * static_cast<T>(input_transform_4x4_3x3[6 * row + i]);
//			return result;
//		}
	};
	template<typename T>
	struct TileTransform<TransformType::OUTPUT, 4, 3, T>
	{
		__device__ Tile<6, T> operator()(const T* src, const int stride, const int row) const noexcept
		{
			return first_transform<4, 6, T>(src, stride, output_transform_4x4_3x3, row);
		}
		__device__ T operator()(const Tile<6, T> &src, const int row) const noexcept
		{
			return second_transform<4, 6, T>(src, output_transform_4x4_3x3, row);
		}
	};
	template<typename T>
	struct TileTransform<TransformType::GRADIENT, 4, 3, T>
	{
		__device__ void operator()(Tile<6, T> &tile) const noexcept
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
		__device__ void operator()(Tile<6, T> &tile) const noexcept
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
	__global__ void kernel_winograd_input_transform2(T* matrices, MatrixShape<TileSize> matrix_shape, const T* input, TensorShape tensor_shape, int2 padding)
	{
		assert(blockDim.x == Elements && blockDim.y == TileSize);
		__shared__ T storage[TileSize * TileSize * Elements];

		for (int filter = threadIdx.x; filter < tensor_shape.filters; filter += blockDim.x)
		{
			for (int row = 0; row < TileSize; row++)
			{
				int batch = blockIdx.x;
				int col = threadIdx.y;
				int x = padding.x + blockIdx.y * TransformSize + row;
				int y = padding.y + blockIdx.z * TransformSize + col;

				SharedArrayShape<TileSize, TileSize, Elements> shared_indexer;
				int storage_offset = shared_indexer.offset_at(row, col, threadIdx.x);
				if (x >= 0 and x < tensor_shape.height and y >= 0 and y < tensor_shape.width)
				{
					int tensor_offset = tensor_shape.offset_at(batch, x, y, filter);
					storage[storage_offset] = input[tensor_offset];
				}
				else
					storage[storage_offset] = zero<T>();
			}
			__syncthreads();

			Tile<TileSize, T> computed_row;
			for (int col = 0; col < TileSize; col++)
			{
				SharedArrayShape<TileSize, TileSize, Elements> shared_indexer;
				int offset = shared_indexer.offset_at(0, col, threadIdx.x);
				int stride = shared_indexer.stride_to_next_row();

				Tile<TileSize, T> loaded_column;
				loaded_column.load(storage, offset, stride);

				TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
				computed_row[col] = transform(loaded_column, threadIdx.y);
			}

			TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
			transform(computed_row);
			int tile_index = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
			int stride = matrix_shape.stride_to_next_column();
			int offset = matrix_shape.offset_at(threadIdx.y, 0, tile_index, filter);
			computed_row.store(matrices, offset, stride);

//			for (int col = 0; col < TileSize; col++)
//			{
//				int tile_index = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
//				int offset = matrix_shape.offset_at(threadIdx.y, col, tile_index, filter);
//				matrices[offset] = 1.0f;
//			}

//			for (int col = 0; col < TileSize; col++)
//			{
//				TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
//				T tmp = transform(computed_row, col);
//
//				int tile_index = (blockIdx.x * gridDim.y + blockIdx.y) * gridDim.z + blockIdx.z;
//				int offset = matrix_shape.offset_at(threadIdx.y, col, tile_index, filter);
//				matrices[offset] = tmp;
//			}
			__syncthreads();
		}
	}

	template<int TransformSize, int KernelSize, unsigned int Elements, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_input_transform3(T* matrices, MatrixShape<TileSize> matrix_shape, const T* input, TensorShape tensor_shape, int2 padding)
	{
		SharedStorage<TileSize, TileSize, Elements, T> storage;

		for (int tile_w = 0; tile_w < tensor_shape.width; tile_w += TransformSize)
		{
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				if (tile_w > 0)
				{
					for (int col = 0; col < (KernelSize - 1); col++)
						storage.at(row, col, threadIdx.x) = storage.at(row, TransformSize + col, threadIdx.x);
				}

				int start_col = (tile_w == 0) ? 0 : (KernelSize - 1);
				for (int col = start_col; col < TileSize; col++)
				{
					int batch = blockIdx.y;
					int x = padding.x + blockIdx.z * TransformSize + row;
					int y = padding.y + tile_w + col;
					int filter = blockIdx.x * blockDim.x + threadIdx.x;

					if (x >= 0 and x < tensor_shape.height and y >= 0 and y < tensor_shape.width)
					{
						int tensor_offset = tensor_shape.offset_at(batch, x, y, filter);
						storage.at(row, col, threadIdx.x) = input[tensor_offset];
					}
					else
						storage.at(row, col, threadIdx.x) = zero<T>();
				}
			}
			__syncthreads();
			if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
			{
				printf("input tile\n");
				for (int row = 0; row < TileSize; row++)
				{
					for (int col = 0; col < TileSize; col++)
						printf("%f ", storage.at(row, col, 0));
					printf("\n");
				}
				printf("\n\n");
			}
			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				Tile<TileSize, T> computed_row;
				for (int col = 0; col < TileSize; col++)
				{
					Tile<TileSize, T> loaded_column = storage.get_column(col, threadIdx.x); // load single column
					TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
					__syncthreads();
					if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
					{
						printf("column %i\n", col);
						for (int i = 0; i < TileSize; i++)
							printf("%f ", loaded_column[i]);
						printf("\n\n");
					}
					__syncthreads();
					computed_row[col] = transform(loaded_column, row); // calculate dot product of given row of transform matrix and loaded column
				}

				__syncthreads();
				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
				{
					for (int i = 0; i < TileSize; i++)
						printf("%f ", computed_row[i]);
					printf("\n");
				}
				__syncthreads();

				int tile_index = tensor_shape.tile_index<TransformSize>(blockIdx.y, blockIdx.z, tile_w / TransformSize);
				int filter = blockIdx.x * blockDim.x + threadIdx.x;
				for (int col = 0; col < TileSize; col++)
				{
					TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
					T tmp = transform(computed_row, col);
					printf("%f ", tmp);
					matrices[matrix_shape.offset_at(row, col, tile_index, filter)] = tmp;
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, int Elements, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_input_transform4(T* matrices, MatrixShape<TileSize> matrix_shape, const T* input, TensorShape input_shape, int2 padding)
	{
		SharedStorage<TileSize, TileSize, Elements, T> storage;

		for (int tile_w = 0; tile_w < input_shape.width; tile_w += TransformSize)
		{
//			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
//			{
//				if (tile_w > 0)
//				{
//					for (int col = 0; col < (KernelSize - 1); col++)
//						storage.at(row, col, threadIdx.x) = storage.at(row, TransformSize + col, threadIdx.x);
//				}
//
//				int filter = blockIdx.x * blockDim.x + threadIdx.x;
//				int x = padding.x + blockIdx.z * TransformSize + row;
//				int offset = input_shape.offset_at(blockIdx.y, x, padding.y + tile_w, filter);
//				int stride = input_shape.stride_to_next_column();
//
//				int start_col = (tile_w == 0) ? 0 : (KernelSize - 1);
//				for (int col = start_col; col < TileSize; col++)
//				{
//					int y = padding.y + tile_w + col;
//
//					if (x >= 0 and x < input_shape.height and y >= 0 and y < input_shape.width)
//						storage.at(row, col, threadIdx.x) = input[offset + col * stride];
//					else
//						storage.at(row, col, threadIdx.x) = zero<T>();
//				}
//			}

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				if (tile_w > 0)
				{
					for (int col = 0; col < (KernelSize - 1); col++)
						storage.at(row, col, threadIdx.x) = storage.at(row, TransformSize + col, threadIdx.x);
				}

				int start_col = (tile_w == 0) ? 0 : (KernelSize - 1);
				for (int col = start_col; col < TileSize; col++)
				{
					int batch = blockIdx.y;
					int x = padding.x + blockIdx.z * TransformSize + row;
					int y = padding.y + tile_w + col;
					int filter = blockIdx.x * blockDim.x + threadIdx.x;

					if (x >= 0 and x < input_shape.height and y >= 0 and y < input_shape.width)
					{
						int tensor_offset = input_shape.offset_at(batch, x, y, filter);
						storage.at(row, col, threadIdx.x) = input[tensor_offset];
					}
					else
						storage.at(row, col, threadIdx.x) = zero<T>();
				}
			}
			__syncthreads();
//			if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//			{
//				printf("input tile\n");
//				for (int row = 0; row < TileSize; row++)
//				{
//					for (int col = 0; col < TileSize; col++)
//						printf("%f ", storage.at(row, col, 0));
//					printf("\n");
//				}
//				printf("\n\n");
//			}
//			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
				Tile<TileSize, T> computed_row = transform(storage.data() + threadIdx.x, Elements, row);

//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
//					printf("computed row\n");
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", computed_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

//				Tile<TileSize, T> final_row;
//				transform.op_transposed(final_row, computed_row);

//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
////					printf("final row\n");
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", final_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

				int tile_index = input_shape.tile_index<TransformSize>(blockIdx.y, blockIdx.z, tile_w / TransformSize);
				int filter = blockIdx.x * blockDim.x + threadIdx.x;
				int offset = matrix_shape.offset_at(row, 0, tile_index, filter);
				int stride = matrix_shape.stride_to_next_column();
				for (int col = 0; col < TileSize; col++)
				{
					T tmp = transform(computed_row, col);
					matrices[offset + col * stride] = tmp;

//					matrices[matrix_shape.offset_at(row, col, tile_index, filter)] = tmp;

//					matrices[offset + col * stride] = final_row[col];
				}
			}
			__syncthreads();
		}
	}

	template<int TransformSize, int KernelSize, unsigned int Elements, class Activation, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_output_transform3(const T* matrices, MatrixShape<TileSize> matrix_shape, T* output, TensorShape tensor_shape, const T* add,
			T alpha1, T alpha2, T beta, const T* bias)
	{
		SharedStorage<TileSize, TileSize, Elements, T> storage;

		for (int filter = threadIdx.x; filter < matrix_shape.filters; filter += blockDim.x)
		{
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
				for (int col = 0; col < TileSize; col++)
				{
					int tile_index = tensor_shape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
					int matrix_offset = matrix_shape.offset_at(row, col, tile_index, filter);
					storage.at(row, col, threadIdx.x) = matrices[matrix_offset];
				}
			__syncthreads();
//			if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//			{
//				printf("input tile\n");
//				for (int row = 0; row < TileSize; row++)
//				{
//					for (int col = 0; col < TileSize; col++)
//						printf("%f ", storage.at(row, col, 0));
//					printf("\n");
//				}
//				printf("\n\n");
//			}
//			__syncthreads();

			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
			{
				Tile<TileSize, T> computed_row;
				TileTransform<TransformType::OUTPUT, TransformSize, KernelSize, T> transform;
				for (int col = 0; col < TransformSize; col++)
				{
					Tile<TileSize, T> loaded_column = storage.get_column(col, threadIdx.x); // load single column
//					__syncthreads();
//					if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//					{
//						printf("column %i\n", col);
//						for (int i = 0; i < TileSize; i++)
//							printf("%f ", loaded_column[i]);
//						printf("\n\n");
//					}
//					__syncthreads();
//					computed_row[col] = transform(loaded_column, row); // calculate dot product of given row of transform matrix and loaded column
				}
//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", computed_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", computed_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;
					if (x < tensor_shape.height and y < tensor_shape.width)
					{
						T tmp = alpha1 * transform(computed_row, col);
						if (bias != nullptr)
							tmp += bias[filter];

						int index = tensor_shape.offset_at(batch, x, y, filter);
						if (add != nullptr)
							tmp += alpha2 * add[index];
						tmp = Activation().forward(tmp);
						if (beta != zero<T>())
							tmp += beta * output[index];
						output[index] = tmp;
					}
				}
			}
			__syncthreads();
		}
	}

	template<int TransformSize, int KernelSize, unsigned int Elements, class Activation, typename T, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_output_transform4(const T* matrices, MatrixShape<TileSize> matrix_shape, T* output, TensorShape output_shape, const T* add,
			T alpha1, T alpha2, T beta, const T* bias)
	{
		SharedStorage<TileSize, TileSize, Elements, T> storage;
		for (int filter = threadIdx.x; filter < matrix_shape.filters; filter += blockDim.x)
		{
			int tile_index = output_shape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				int offset = matrix_shape.offset_at(row, 0, tile_index, filter);
				int stride = matrix_shape.stride_to_next_column();
				for (int col = 0; col < TileSize; col++)
					storage.at(row, col, threadIdx.x) = matrices[offset + col * stride];
			}
//			__syncthreads();
//			if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//			{
//				printf("input tile\n");
//				for (int row = 0; row < TileSize; row++)
//				{
//					for (int col = 0; col < TileSize; col++)
//						printf("%f ", storage.at(row, col, 0));
//					printf("\n");
//				}
//				printf("\n\n");
//			}
			__syncthreads();

			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
			{
				TileTransform<TransformType::OUTPUT, TransformSize, KernelSize, T> transform;
				Tile<TileSize, T> computed_row = transform(storage.data() + threadIdx.x, Elements, row);

//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
////					printf("computed row\n");
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", computed_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

//				__syncthreads();
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//				{
//					//					printf("final row\n");
//					for (int i = 0; i < TileSize; i++)
//						printf("%f ", final_row[i]);
//					printf("\n");
//				}
//				__syncthreads();

				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;
					if (x < output_shape.height and y < output_shape.width)
					{
						T tmp = alpha1 * transform(computed_row, col);

//						if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//							printf("%f ", tmp);

						if (bias != nullptr)
							tmp += bias[filter];
						int index = output_shape.offset_at(batch, x, y, filter);
						if (add != nullptr)
							tmp += alpha2 * add[index];
						tmp = Activation().forward(tmp);
						if (beta != zero<T>())
							tmp += beta * output[index];
						output[index] = tmp;
					}
				}
//				if (blockIdx.x == 0 and blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and threadIdx.y == 0)
//					printf("\n");
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

//	template<int TransformSize, int KernelSize, unsigned int Elements, typename T, int TileSize = TransformSize + KernelSize - 1>
//	__global__ void kernel_winograd_input_transform(T* matrices, MatrixShape<TileSize> matrix_shape, const T* input, TensorShape tensor_shape, int2 padding)
//	{
//		assert(blockDim.x == Elements && blockDim.y == 1);
//		__shared__ T storage[TileSize * TileSize * Elements];
//
//		for (unsigned int filter = blockIdx.x * blockDim.x + threadIdx.x; filter < tensor_shape.filters; filter += gridDim.x * blockDim.x)
//		{
//			for (unsigned int column = 0; column < KernelSize; column++)
//			{
//				int offset, stride;
//				offset = tensor_shape.offset_at(blockIdx.x, padding.x+, column, filter);
//				stride = tensor_shape.stride_to_next_row();
//				Tile<TileSize, T> tile;
//				tile.load(input, offset, stride);
//
//				TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
//				Tile<TileSize, T> transformed = transform(tile);
//
//				SharedArrayShape<TileSize, TileSize, Elements> shared_indexer;
//				offset = shared_indexer.offset_at(0, column, threadIdx.x);
//				stride = shared_indexer.stride_to_next_row();
//				transformed.store(storage, offset, stride);
//			}
//
//			for (unsigned int row = 0; row < TileSize; row++)
//			{
//				SharedArrayShape<TileSize, TileSize, Elements> shared_indexer;
//				int offset = shared_indexer.offset_at(row, 0, threadIdx.x);
//				int stride = shared_indexer.stride_to_next_column();
//
//				Tile<TileSize, T> tile;
//				tile.load(storage, offset, stride);
//
//				TileTransform<TransformType::INPUT, TransformSize, KernelSize, T> transform;
//				Tile<TileSize, T> transformed = transform(tile);
//
//				int tile_index = blockIdx.x * blockIdx.y * blockIdx.z;
//				offset = matrix_shape.offset_at(row, 0, tile_index, filter);
//				stride = matrix_shape.stride_to_next_column();
//				transformed.store(matrices, offset, stride);
//			}
//		}
//	}

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

	template<int TransformSize, int KernelSize, typename T>
	void setup_transform_matrix()
	{

	}
}

namespace avocado
{
	namespace backend
	{

		avSize_t winogradGetWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc)
		{
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
//			{
//				int3 shape { getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//
//				int tiles_h = (getTensor(xDesc).dimension(1) + 3) / 4;
//				int tiles_w = (getTensor(xDesc).dimension(2) + 3) / 4;
//				dim3 gridSize(getTensor(xDesc).firstDim(), tiles_h, tiles_w);
//				cudaStream_t stream = getContext(context).getStream();
//
//				dim3 blockSize(32, 6);
//				kernel_conv3x3_4x4_input_transform<64> <<<gridSize, blockSize, 0, stream>>>(getPointer<float>(matricesMem), getPointer<float>(xMem), shape);
//				cudaStreamSynchronize(stream);
//			}
//
//			{
//				uint4 shape { getTensor(xDesc).dimension(0), getTensor(xDesc).dimension(1), getTensor(xDesc).dimension(2), getTensor(xDesc).dimension(3) };
//				int2 padding { -1, -1 };
//				dim3 blockDim(128);
//				dim3 gridDim(shape.x, (shape.y + 3) / 4, (shape.z + 3) / 4);
//				cudaStream_t stream = getContext(context).getStream();
//
//				switch (getTensor(xDesc).dtype())
//				{
//					case AVOCADO_DTYPE_FLOAT32:
//						kernel_winograd_input_transform<4, 3, 128> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(matricesMem), getPointer<float>(xMem),
//								shape, padding, 0.0f);
//						break;
//				}
//				cudaStreamSynchronize(stream);
//			}
//
//			{
//				TensorShape tensor_shape = get_tensor_shape(getTensor(xDesc));
//				MatrixShape<6> matrix_shape = get_matrix_shape<6>(getTensor(matricesDesc));
//
//				int filters_out = getTensor(xDesc).firstDim();
//				int filters_in = getTensor(xDesc).lastDim();
//
//				int batch_size = getTensor(xDesc).dimension(0);
//				int tile_h = (getTensor(xDesc).dimension(1) + 3) / 4;
//				int tile_w = (getTensor(xDesc).dimension(2) + 3) / 4;
//
//				int2 padding { -1, -1 };
//
//				dim3 blockDim(64, 6);
//				dim3 gridDim(batch_size, tile_h, tile_w);
//				cudaStream_t stream = getContext(context).getStream();
//
//				switch (getTensor(xDesc).dtype())
//				{
//					case AVOCADO_DTYPE_FLOAT32:
//						kernel_winograd_input_transform2<4, 3, 64> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(matricesMem), matrix_shape,
//								getPointer<float>(xMem), tensor_shape, padding);
//						break;
//				}
//				cudaStreamSynchronize(stream);
//			}
//
//			{
//				TensorShape tensor_shape = get_tensor_shape(getTensor(xDesc));
//				MatrixShape<6> matrix_shape = get_matrix_shape<6>(getTensor(matricesDesc));
//
//				int batch_size = getTensor(xDesc).dimension(0);
//				int tile_h = (getTensor(xDesc).dimension(1) + 3) / 4;
//				int tile_w = (getTensor(xDesc).dimension(2) + 3) / 4;
//				int filters_in = getTensor(xDesc).dimension(3);
//
//				int2 padding { -1, -1 };
//
//				dim3 blockDim(128, 1);
//				dim3 gridDim((filters_in + blockDim.x - 1) / blockDim.x, batch_size, tile_h);
//				cudaStream_t stream = getContext(context).getStream();
//
//				switch (getTensor(xDesc).dtype())
//				{
//					case AVOCADO_DTYPE_FLOAT32:
//						kernel_winograd_input_transform3<4, 3, 128> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
//								getPointer<float>(matricesMem), matrix_shape, getPointer<float>(xMem), tensor_shape, padding);
//						break;
//				}
//				cudaStreamSynchronize(stream);
//			}

			{
				float matrix[] = { 1.0f, 0.0f, -1.25f, 0.0f, 0.25f, 0.0f, 0.0f, 1.0f, 1.0f, -0.25f, -0.25f, 0.0f, 0.0f, -1.0f, 1.0f, 0.25f, -0.25f, 0.0f, 0.0f,
						-1.0f, -0.5f, 1.0f, 0.5f, 0.0f, 0.0f, 1.0f, -0.5f, -1.0f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, -1.25f, 0.0f, 0.25f };

				TensorShape tensor_shape = get_tensor_shape(getTensor(xDesc));
				MatrixShape<6> matrix_shape = get_matrix_shape<6>(getTensor(matricesDesc));

				int batch_size = getTensor(xDesc).dimension(0);
				int tile_h = (getTensor(xDesc).dimension(1) + 3) / 4;
				int tile_w = (getTensor(xDesc).dimension(2) + 3) / 4;
				int filters_in = getTensor(xDesc).dimension(3);

				int2 padding { -1, -1 };

				dim3 blockDim(128, 3);
				dim3 gridDim((filters_in + blockDim.x - 1) / blockDim.x, batch_size, tile_h);
				cudaStream_t stream = getContext(context).getStream();

				cudaMemcpyToSymbolAsync(input_transform_4x4_3x3, &matrix, 6 * 6 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);
				switch (getTensor(xDesc).dtype())
				{
					case AVOCADO_DTYPE_FLOAT32:
						kernel_winograd_input_transform4<4, 3, 128> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
								getPointer<float>(matricesMem), matrix_shape, getPointer<float>(xMem), tensor_shape, padding);
						break;
				}
				cudaStreamSynchronize(stream);
			}
			return AVOCADO_STATUS_NOT_SUPPORTED;
		}

		avStatus_t winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation)
		{
			TensorShape tensor_shape = get_tensor_shape(getTensor(yDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(getTensor(matricesDesc));

			int batch_size = getTensor(yDesc).dimension(0);
			int tile_h = (getTensor(yDesc).dimension(1) + 3) / 4;
			int tile_w = (getTensor(yDesc).dimension(2) + 3) / 4;
			int filters_in = getTensor(yDesc).dimension(3);

			dim3 blockDim(128, 4);
			dim3 gridDim(batch_size, tile_h, tile_w);
			cudaStream_t stream = getContext(context).getStream();

			float matrix[] = { 1.0f, 1.0f, 1.0f, 0.25f, 0.25f, 0.0f, 0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
					-1.0f, 2.0f, -2.0f, 2.0f };
			cudaMemcpyToSymbolAsync(output_transform_4x4_3x3, &matrix, 4 * 6 * sizeof(float), 0, cudaMemcpyHostToDevice, stream);

			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					float _alpha1 = getAlphaValue(alpha1);
					float _alpha2 = getAlphaValue(alpha2);
					float _beta = getBetaValue(beta);
					kernel_winograd_output_transform4<4, 3, 128, ActivationRelu<float>> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
							getPointer<float>(matricesMem), matrix_shape, getPointer<float>(yMem), tensor_shape, getPointer<float>(zMem), _alpha1, _alpha2,
							_beta, getPointer<float>(bMem));
				}
					break;
			}
			cudaStreamSynchronize(stream);
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

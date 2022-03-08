/*
 * winograd_nonfused.cu
 *
 *  Created on: Dec 29, 2021
 *      Author: Maciej Kozarzewski
 */

#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>

#include "activations.cuh"
#include "utilities.hpp"
#include "numbers/numbers.cuh"

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cassert>
#include <iostream>
#include <array>

namespace
{
	using namespace avocado::backend;
	using namespace avocado::backend::cuda;
	using namespace numbers;

	enum class TransformType
	{
		WEIGHT, INPUT, OUTPUT, GRADIENT, UPDATE
	};

	template<TransformType Type, int TransformSize, int KernelSize>
	__host__ __device__ constexpr int transform_size()
	{
		constexpr int TileSize = TransformSize + KernelSize - 1;
		switch (Type)
		{
			case TransformType::WEIGHT:
				return TileSize * KernelSize;
			case TransformType::INPUT:
				return TileSize * TileSize;
			case TransformType::OUTPUT:
				return TransformSize * TileSize;
			case TransformType::GRADIENT:
				return TileSize * TransformSize;
			case TransformType::UPDATE:
				return KernelSize * TileSize;
		}
		return 0;
	}
	template<TransformType Type, int TransformSize, int KernelSize>
	__host__ __device__ constexpr int transform_offset()
	{
		constexpr int TileSize = TransformSize + KernelSize - 1;
		switch (Type)
		{
			case TransformType::WEIGHT:
				return 0;
			case TransformType::INPUT:
				return TileSize * KernelSize;
			case TransformType::OUTPUT:
				return TileSize * KernelSize + TileSize * TileSize;
			case TransformType::GRADIENT:
				return TileSize * KernelSize + TileSize * TileSize + TransformSize * TileSize;
			case TransformType::UPDATE:
				return TileSize * KernelSize + TileSize * TileSize + TransformSize * TileSize + TileSize * TransformSize;
		}
		return 0;
	}
	template<int TransformSize, int KernelSize>
	__host__ __device__ constexpr int total_transform_size()
	{
		return transform_size<TransformType::WEIGHT, TransformSize, KernelSize>() + transform_size<TransformType::INPUT, TransformSize, KernelSize>()
				+ transform_size<TransformType::OUTPUT, TransformSize, KernelSize>() + transform_size<TransformType::GRADIENT, TransformSize, KernelSize>()
				+ transform_size<TransformType::UPDATE, TransformSize, KernelSize>();
	}

	__constant__ float float_transforms_2x2_3x3[total_transform_size<2, 3>()];
	__constant__ float float_transforms_4x4_3x3[total_transform_size<4, 3>()];
	__constant__ float float_transforms_2x2_5x5[total_transform_size<2, 5>()];

	template<TransformType Type, int TransformSize, int KernelSize>
	struct TileTransform
	{
	};
	template<TransformType Type>
	struct TileTransform<Type, 4, 3>
	{
		__host__ __device__ const float* get() const
		{
			return float_transforms_4x4_3x3 + transform_offset<Type, 4, 3>();
		}
	};
	template<TransformType Type>
	struct TileTransform<Type, 2, 3>
	{
		__host__ __device__ const float* get() const
		{
			return float_transforms_2x2_3x3 + transform_offset<Type, 2, 3>();
		}
	};
	template<TransformType Type>
	struct TileTransform<Type, 2, 5>
	{
		__host__ __device__ const float* get() const
		{
			return float_transforms_2x2_5x5 + transform_offset<Type, 2, 5>();
		}
	};

	struct TensorShape
	{
		int batch = 0;
		int height = 0;
		int width = 0;
		int filters = 0;

		__device__ TensorShape() = default;
		__host__ TensorShape(const TensorDescriptor &desc) :
				batch(desc.dimension(0)), height(desc.dimension(1)), width(desc.dimension(2)), filters(desc.dimension(3))
		{
		}

		__device__ int offset_at(int b, int h, int w, int f) const
		{
			assert(b >= 0 && b < batch);
			assert(h >= 0 && h < height);
			assert(w >= 0 && w < width);
			assert(f >= 0 && f < filters);
			return ((b * height + h) * width + w) * filters + f;
		}
		template<int TileSize>
		__device__ int tile_index(int b, int h, int w) const
		{
			assert(b >= 0 && b < batch);
			assert(h >= 0 && h < tiles_vertically<TileSize>());
			assert(w >= 0 && w < tiles_horizontally<TileSize>());
			return (b * tiles_vertically<TileSize>() + h) * tiles_horizontally<TileSize>() + w;
		}
		template<int TileSize>
		__device__ int tiles_vertically() const
		{
			return (height + TileSize - 1) / TileSize;
		}
		template<int TileSize>
		__device__ int tiles_horizontally() const
		{
			return (width + TileSize - 1) / TileSize;
		}
	};
	struct MatrixShape
	{
		int tile_size = 0;
		int nb_tiles = 0;
		int filters = 0;

		__device__ MatrixShape() = default;
		__host__ MatrixShape(const TensorDescriptor &desc) :
				tile_size(sqrt(desc.dimension(0))), nb_tiles(desc.dimension(1)), filters(desc.dimension(2))
		{
		}

		__device__ int offset_at(int r, int c, int t, int f) const
		{
			return ((r * tile_size + c) * nb_tiles + t) * filters + f;
		}
		__device__ int stride_to_next_row() const
		{
			return tile_size * nb_tiles * filters;
		}
		__device__ int stride_to_next_column() const
		{
			return nb_tiles * filters;
		}
	};

	template<int Size, typename T>
	struct Tile
	{
	private:
		Number<T> data[Size];
	public:
		__device__ void load(const T* src, int offset, int stride, int elementsToLoad)
		{
			for (int i = 0; i < Size; i++)
				data[i].load(src + offset + i * stride, elementsToLoad);
		}
		__device__ void load(const Number<T>* src, int offset, int stride)
		{
			for (int i = 0; i < Size; i++)
				data[i] = src[offset + i * stride];
		}
		__device__ void store(T* src, int offset, int stride, int elementsToStore) const
		{
			for (int i = 0; i < Size; i++)
				data[i].store(src + offset + i * stride, elementsToStore);
		}
		__device__ void store(Number<T>* src, int offset, int stride) const
		{
			for (int i = 0; i < Size; i++)
				src[offset + i * stride] = data[i];
		}
		__device__ Number<T>& operator[](int index)
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		__device__ Number<T> operator[](int index) const
		{
			assert(index >= 0 && index < Size);
			return data[index];
		}
		__device__ Tile<Size, Number<T>>& operator+=(Number<T> x)
		{
			for (int i = 0; i < Size; i++)
				data[i] += x;
			return *this;
		}
	};

	template<int Rows, int Columns, int Elements, typename T>
	struct SharedStorageWrapper
	{
	private:
		Number<T> *m_data;
	public:

		__device__ SharedStorageWrapper(Number<T> *ptr) :
				m_data(ptr)
		{
		}
		__device__ Number<T>* data()
		{
			return m_data;
		}
		__device__ int offset_at(int row, int col, int element) const
		{
			assert(row >= 0 && row < Rows);
			assert(col >= 0 && col < Columns);
			assert(element >= 0 && element < Elements);
			return (row * Columns + col) * Elements + element;
		}
		__device__ Tile<Columns, Number<T>> get_row(int rowIndex, int elementOffset)
		{
			Tile<Columns, Number<T>> result;
			for (int col = 0; col < Columns; col++)
				result[col] = m_data[offset_at(rowIndex, col, elementOffset)];
			return result;
		}
		__device__ Tile<Rows, Number<T>> get_column(int columnIndex, int elementOffset)
		{
			Tile<Columns, Number<T>> result;
			for (int row = 0; row < Columns; row++)
				result[row] = m_data[offset_at(row, columnIndex, elementOffset)];
			return result;
		}
		__device__ Number<T>& at(int row, int col, int element)
		{
			return m_data[offset_at(row, col, element)];
		}
		__device__ Number<T> at(int row, int col, int element) const
		{
			return m_data[offset_at(row, col, element)];
		}
		__device__ int stride_to_next_row() const
		{
			return Columns * Elements;
		}
		__device__ int stride_to_next_column() const
		{
			return Elements;
		}

	};

	/**
	 * \brief Performs the first part of Winograd transform.
	 *
	 * \param[in] src Pointer to memory containing tile data.
	 * \param[in] stride Number of elements between subsequent columns of tile data
	 * \param[in] transformMatrix Pointer to array of transform coefficients.
	 * \param[in] row Selects which row of transform matrix should be multiplied by tile data.
	 */
	template<int Rows, int Columns, typename T>
	__device__ Tile<Columns, T> first_transform(const Number<T>* src, const int stride, const Number<T>* transformMatrix, const int row)
	{
		assert(row >= 0 && row < Rows);
		Tile<Columns, T> result;
		for (int i = 0; i < Columns; i++)
			result[i] = numbers::zero<T>();
		int index = 0;
		for (int col = 0; col < Columns; col++)
		{
			Number<T> c = transformMatrix[row * Columns + col];
			if (c != numbers::zero<T>())
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
	__device__ Number<T> second_transform(const Tile<Columns, T> &src, const Number<T> *transformMatrix, const int row)
	{
		assert(row >= 0 && row < Rows);
		Number<T> result = numbers::zero<T>();
		for (int i = 0; i < Columns; i++)
			result += src[i] * transformMatrix[row * Columns + i];
		return result;
	}

	template<int TransformSize, int KernelSize, int Elements, typename T>
	__global__ void kernel_winograd_weight_transform(T* matrices, MatrixShape matrixShape, const T* weights, TensorShape weightsShape, bool invert)
	{
		assert(blockDim.x == Elements);
		constexpr int TileSize = TransformSize + KernelSize - 1;

		__shared__ Number<T> transform_matrix[TileSize * KernelSize];
		for (int i = threadIdx.x; i < TileSize * KernelSize; i += blockDim.x) // load transform matrix
			transform_matrix[i] = TileTransform<TransformType::WEIGHT, TransformSize, KernelSize>().get()[i];
		__syncthreads();

		__shared__ Number<T> shared_storage[KernelSize * KernelSize * Elements];
		SharedStorageWrapper<KernelSize, KernelSize, Elements, T> storage_wrapper(shared_storage);

		for (int filter = length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); filter < weightsShape.filters; filter += length<T>() * blockDim.x * gridDim.x)
		{
			for (int i = threadIdx.y; i < square(KernelSize); i += blockDim.y)
			{
				int row = i / KernelSize;
				int col = i % KernelSize;
				int tensor_offset = weightsShape.offset_at(blockIdx.y, row, col, filter);
				int elements_to_load = weightsShape.filters - filter;
				Number<T> tmp(weights + tensor_offset, elements_to_load);
				if (invert)
					storage_wrapper.at(KernelSize - 1 - row, KernelSize - 1 - col, threadIdx.x) = tmp;
				else
					storage_wrapper.at(row, col, threadIdx.x) = tmp;
			}
			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				Tile<KernelSize, T> computed_row = first_transform<TileSize, KernelSize>(storage_wrapper.data() + threadIdx.x, Elements, transform_matrix, row);

				int offset = matrixShape.offset_at(row, 0, blockIdx.y, filter);
				int stride = matrixShape.stride_to_next_column();
				int elements_to_store = weightsShape.filters - filter;
				for (int col = 0; col < TileSize; col++)
				{
					Number<T> tmp = second_transform<TileSize, KernelSize>(computed_row, transform_matrix, col);
					tmp.store(matrices + offset + col * stride, elements_to_store);
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, int Elements, typename T>
	__global__ void kernel_winograd_input_transform(T* matrices, MatrixShape matrixShape, const T* input, TensorShape inputShape, int2 padding,
			const T paddingValue)
	{
		assert(blockDim.x == Elements);
		constexpr int TileSize = TransformSize + KernelSize - 1;

		__shared__ Number<T> transform_matrix[TileSize * TileSize];
		for (int i = threadIdx.x; i < TileSize * TileSize; i += blockDim.x) // load transform matrix
			transform_matrix[i] = TileTransform<TransformType::INPUT, TransformSize, KernelSize>().get()[i];
		__syncthreads();

		__shared__ Number<T> shared_storage[TileSize * TileSize * Elements];
		SharedStorageWrapper<TileSize, TileSize, Elements, T> storage_wrapper(shared_storage);

		for (int tile_w = 0; tile_w < inputShape.width; tile_w += TransformSize)
		{
			int filter = length<T>() * (blockIdx.x * blockDim.x + threadIdx.x);
			if (filter < inputShape.filters)
				for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
				{
					if (tile_w > 0)
					{
						for (int col = 0; col < (KernelSize - 1); col++)
							storage_wrapper.at(row, col, threadIdx.x) = storage_wrapper.at(row, TransformSize + col, threadIdx.x);
					}

					int start_col = (tile_w == 0) ? 0 : (KernelSize - 1);
					for (int col = start_col; col < TileSize; col++)
					{
						int batch = blockIdx.y;
						int x = padding.x + blockIdx.z * TransformSize + row;
						int y = padding.y + tile_w + col;

						if (x >= 0 and x < inputShape.height and y >= 0 and y < inputShape.width)
						{
							int tensor_offset = inputShape.offset_at(batch, x, y, filter);
							int elements_to_load = inputShape.filters - filter;
							storage_wrapper.at(row, col, threadIdx.x).load(input + tensor_offset, elements_to_load);
						}
						else
							storage_wrapper.at(row, col, threadIdx.x) = Number<T>(paddingValue);
					}
				}
			__syncthreads();

			if (filter < inputShape.filters)
				for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
				{
					Tile<TileSize, T> computed_row = first_transform<TileSize, TileSize>(storage_wrapper.data() + threadIdx.x, Elements, transform_matrix, row);

					int tile_index = inputShape.tile_index<TransformSize>(blockIdx.y, blockIdx.z, tile_w / TransformSize);
					int offset = matrixShape.offset_at(row, 0, tile_index, filter);
					int stride = matrixShape.stride_to_next_column();
					for (int col = 0; col < TileSize; col++)
					{
						Number<T> tmp = second_transform<TileSize, TileSize>(computed_row, transform_matrix, col);
						int elements_to_store = inputShape.filters - filter;
						tmp.store(matrices + offset + col * stride, elements_to_store);
					}
				}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, unsigned int Elements, typename T, typename U = T>
	__global__ void kernel_winograd_output_transform(const T* matrices, MatrixShape matrixShape, T* output, TensorShape outputShape, const T* add, U alpha1,
			U alpha2, U beta, const U* bias, avActivationType_t activation)
	{
		assert(blockDim.x == Elements);
		constexpr int TileSize = TransformSize + KernelSize - 1;

		__shared__ Number<T> transform_matrix[TileSize * TileSize];
		for (int i = threadIdx.x; i < TileSize * TileSize; i += blockDim.x) // load transform matrix
			transform_matrix[i] = TileTransform<TransformType::OUTPUT, TransformSize, KernelSize>().get()[i];
		__syncthreads();

		__shared__ Number<T> shared_storage[TileSize * TileSize * Elements];
		SharedStorageWrapper<TileSize, TileSize, Elements, T> storage_wrapper(shared_storage);

		for (int filter = length<T>() * threadIdx.x; filter < matrixShape.filters; filter += length<T>() * blockDim.x)
		{
			int tile_index = outputShape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				int offset = matrixShape.offset_at(row, 0, tile_index, filter);
				int stride = matrixShape.stride_to_next_column();
				int elements_to_load = matrixShape.filters - filter;
				for (int col = 0; col < TileSize; col++)
					storage_wrapper.at(row, col, threadIdx.x).load(matrices + offset + col * stride, elements_to_load);
			}
			__syncthreads();

			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
			{
				Tile<TileSize, T> computed_row = first_transform<TransformSize, TileSize>(storage_wrapper.data() + threadIdx.x, Elements, transform_matrix,
						row);

				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;
					int elements_to_store = matrixShape.filters - filter;
					if (x < outputShape.height and y < outputShape.width)
					{
						Number<T> tmp = Number<T>(alpha1) * second_transform<TransformSize, TileSize>(computed_row, transform_matrix, col);

						if (bias != nullptr)
							tmp += Number<T>(bias + filter, elements_to_store);
						int index = outputShape.offset_at(batch, x, y, filter);
						if (add != nullptr)
							tmp += Number<T>(alpha2) * Number<T>(add + index, elements_to_store);
						tmp = activation_forward(activation, tmp);
						if (beta != scalar_zero<U>())
							tmp += Number<T>(beta) * Number<T>(output + index, elements_to_store);
						tmp.store(output + index, elements_to_store);
					}
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, int Elements, typename T>
	__global__ void kernel_winograd_gradient_transform(T* matrices, MatrixShape matrixShape, const T* gradient, TensorShape gradientShape)
	{
		assert(blockDim.x == Elements);
		constexpr int TileSize = TransformSize + KernelSize - 1;

		__shared__ Number<T> transform_matrix[TileSize * TransformSize];
		for (int i = threadIdx.x; i < TileSize * TransformSize; i += blockDim.x) // load transform matrix
			transform_matrix[i] = TileTransform<TransformType::GRADIENT, TransformSize, KernelSize>().get()[i];
		__syncthreads();

		__shared__ Number<T> shared_storage[TransformSize * TransformSize * Elements];
		SharedStorageWrapper<TransformSize, TransformSize, Elements, T> storage_wrapper(shared_storage);

		for (int filter = length<T>() * threadIdx.x; filter < gradientShape.filters; filter += length<T>() * blockDim.x)
		{
			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;

					if (x < gradientShape.height and y < gradientShape.width)
					{
						int tensor_offset = gradientShape.offset_at(batch, x, y, filter);
						int elements_to_load = gradientShape.filters - filter;
						storage_wrapper.at(row, col, threadIdx.x).load(gradient + tensor_offset, elements_to_load);
					}
					else
						storage_wrapper.at(row, col, threadIdx.x) = numbers::zero<T>();
				}
			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				Tile<TransformSize, T> computed_row = first_transform<TileSize, TransformSize>(storage_wrapper.data() + threadIdx.x, Elements, transform_matrix,
						row);

				int tile_index = gradientShape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
				int offset = matrixShape.offset_at(row, 0, tile_index, filter);
				int stride = matrixShape.stride_to_next_column();
				int elements_to_store = gradientShape.filters - filter;
				for (int col = 0; col < TileSize; col++)
				{
					Number<T> tmp = second_transform<TileSize, TransformSize>(computed_row, transform_matrix, col);
					tmp.store(matrices + offset + col * stride, elements_to_store);
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, unsigned int Elements, typename T>
	__global__ void kernel_winograd_update_transform(const T* matrices, MatrixShape matrixShape, T* update, TensorShape updateShape, T alpha, T beta)
	{
		assert(blockDim.x == Elements);
		constexpr int TileSize = TransformSize + KernelSize - 1;

		__shared__ Number<T> transform_matrix[KernelSize * TileSize];
		for (int i = threadIdx.x; i < KernelSize * TileSize; i += blockDim.x) // load transform matrix
			transform_matrix[i] = TileTransform<TransformType::UPDATE, TransformSize, KernelSize>().get()[i];
		__syncthreads();

		__shared__ Number<T> shared_storage[TileSize * TileSize * Elements];
		SharedStorageWrapper<TileSize, TileSize, Elements, T> storage_wrapper(shared_storage);

		for (int filter = length<T>() * (blockIdx.x * blockDim.x + threadIdx.x); filter < matrixShape.filters; filter += length<T>() * blockDim.x * gridDim.x)
		{
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				int filter_out = blockIdx.y;
				int offset = matrixShape.offset_at(row, 0, filter_out, filter);
				int stride = matrixShape.stride_to_next_column();
				int elements_to_load = matrixShape.filters - filter;
				for (int col = 0; col < TileSize; col++)
					storage_wrapper.at(row, col, threadIdx.x).load(matrices + offset + col * stride, elements_to_load);
			}
			__syncthreads();

			for (int row = threadIdx.y; row < KernelSize; row += blockDim.y)
			{
				Tile<TileSize, T> computed_row = first_transform<KernelSize, TileSize>(storage_wrapper.data() + threadIdx.x, Elements, transform_matrix, row);

				for (int col = 0; col < KernelSize; col++)
				{
					Number<T> tmp = Number<T>(alpha) * second_transform<KernelSize, TileSize>(computed_row, transform_matrix, col);

					int filter_out = blockIdx.y;
					int index = updateShape.offset_at(filter_out, row, col, filter);
					int elements_to_store = matrixShape.filters - filter;
					if (beta != scalar_zero<T>())
						tmp += Number<T>(beta) * Number<T>(update + index, elements_to_store);
					tmp.store(update + index, elements_to_store);
				}
			}
			__syncthreads();
		}
	}

	template<typename T>
	size_t bytes(const std::vector<T> &a) noexcept
	{
		return sizeof(T) * a.size();
	}

	struct TransformSetup
	{
	private:
		static void setup_2x2_3x3()
		{
			std::vector<float> transform = {
			/* weight */1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1.0,
			/* input */1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
			/* output */1.0, 0.5, 0.5, 0.0, 0.0, 0.5, -0.5, 1.0,
			/* gradient */1.0, 0.0, 1.0, 1.0, 1.0, -1.0, 0.0, 1.0,
			/* update */1.0, 0.5, 0.5, 0.0, 0.0, 0.5, -0.5, 0.0, 0.0, 0.5, 0.5, 1.0 };

			if (bytes(transform) != sizeof(float) * total_transform_size<2, 3>())
				throw std::runtime_error("incorrect number of bytes in the transform 2x2 3x3 matrix");
			cudaError_t error = cudaMemcpyToSymbol(float_transforms_2x2_3x3, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
				throw std::runtime_error("transform matrices setup failed");
		}
		static void setup_4x4_3x3()
		{
			std::vector<float> transform = {
			/* weight */1.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, -2.0
					/ 3.0, 4.0 / 3.0, 0.0, 0.0, 2.0
			/* input */, 1.0, 0.0, -1.25, 0.0, 0.25, 0.0, 0.0, 1.0, 1.0, -0.25, -0.25, 0.0, 0.0, -1.0, 1.0, 0.25, -0.25, 0.0, 0.0, -1.0, -0.5, 1.0, 0.5, 0.0,
					0.0, 1.0, -0.5, -1.0, 0.5, 0.0, 0.0, 1.0, 0.0, -1.25, 0.0, 0.25
					/* output */, 1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 2.0
					/* gradient */, 1.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0
							/ 3.0, 4.0 / 3.0, 8.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -8.0 / 3.0, 0.0, 0.0, 0.0, 2.0
					/*  * update */, 1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0 };

			if (bytes(transform) != sizeof(float) * total_transform_size<4, 3>())
				throw std::runtime_error("incorrect number of bytes in the transform 4x4 3x3 matrix");
			cudaError_t error = cudaMemcpyToSymbol(float_transforms_4x4_3x3, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
				throw std::runtime_error("transform matrices setup failed");
		}
		static void setup_2x2_5x5()
		{
			std::vector<float> transform = {
			/* weight */1.0, 0.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0,
					1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, 1.0 / 6.0, -1.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0, 8.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 2.0,
					/* input */1.0, 0.0, -1.25, 0.0, 0.25, 0.0, 0.0, 1.0, 1.0, -0.25, -0.25, 0.0, 0.0, -1.0, 1.0, 0.25, -0.25, 0.0, 0.0, -1.0, -0.5, 1.0, 0.5,
					0.0, 0.0, 1.0, -0.5, -1.0, 0.5, 0.0, 0.0, 1.0, 0.0, -1.25, 0.0, 0.25,
					/* output */1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 2.0,
					/* gradient */1.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 0.0, 1.0,
					/* update */1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 0.0,
					0.0, 1.0, 1.0, 4.0, 4.0, 4.0 };

			if (bytes(transform) != sizeof(float) * total_transform_size<2, 5>())
				throw std::runtime_error("incorrect number of bytes in the transform 2x2 5x5 matrix");
			cudaError_t error = cudaMemcpyToSymbol(float_transforms_2x2_5x5, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);
			if (error != cudaSuccess)
				throw std::runtime_error("transform matrices setup failed");
		}
	public:
		static avStatus_t setup()
		{
			static const bool already_done = []()
			{
				try
				{
					setup_2x2_3x3();
					setup_4x4_3x3();
					setup_2x2_5x5();
					return true;
				}
				catch(std::exception &e)
				{
					std::cout<<e.what()<<'\n';
					return false;
				}
			}();
			if (already_done)
				return AVOCADO_STATUS_SUCCESS;
			else
				return AVOCADO_STATUS_INTERNAL_ERROR;
		}
	};

	template<typename T>
	avStatus_t launch_weight_transform(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &wDesc,
			const MemoryDescriptor &wMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem, int transformSize)
	{
		context.setDevice();
		TensorShape tensor_shape(wDesc);
		MatrixShape matrix_shape(matricesDesc);

		const bool invert = (config.mode == AVOCADO_CROSS_CORRELATION_MODE);
		const int filter_size = wDesc.dimension(1);
		const int filters_out = wDesc.firstDim();
		const int filters_in = wDesc.lastDim();

		constexpr int Elements = std::is_same<T, double>::value ? 64 : 128;
		dim3 blockDim(Elements, 1);
		dim3 gridDim(gridSize<32>(filters_in, blockDim.x), gridSize<1024>(filters_out, blockDim.y));
		cudaStream_t stream = context.getStream();

		if (is_conv(3, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_weight_transform<2, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, wMem.data<T>(),
							tensor_shape, invert);
					return checkForErrors();
				case 4:
					kernel_winograd_weight_transform<4, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, wMem.data<T>(),
							tensor_shape, invert);
					return checkForErrors();
			}
		}
		if (is_conv(5, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_weight_transform<2, 5, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, wMem.data<T>(),
							tensor_shape, invert);
					return checkForErrors();
			}
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	template<typename T>
	avStatus_t launch_input_transform(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &wDesc,
			const TensorDescriptor &xDesc, const MemoryDescriptor &xMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem, int transformSize)
	{
		context.setDevice();
		TensorShape tensor_shape(xDesc);
		MatrixShape matrix_shape(matricesDesc);

		const int filter_size = wDesc.dimension(1);

		const int batch_size = xDesc.dimension(0);
		const int tile_h = (xDesc.dimension(1) + transformSize - 1) / transformSize;
		const int tile_w = (xDesc.dimension(2) + transformSize - 1) / transformSize;
		const int filters_in = xDesc.dimension(3);

		int2 padding { config.padding[0], config.padding[1] };
		T padding_value = config.getPaddingValue<T>();

		constexpr int Elements = std::is_same<T, double>::value ? 64 : 128;
		dim3 blockDim(Elements, filter_size);
		dim3 gridDim((filters_in + blockDim.x - 1) / blockDim.x, batch_size, tile_h);
		cudaStream_t stream = context.getStream();

		if (is_conv(3, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_input_transform<2, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, xMem.data<T>(),
							tensor_shape, padding, padding_value);
					return checkForErrors();
				case 4:
					kernel_winograd_input_transform<4, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, xMem.data<T>(),
							tensor_shape, padding, padding_value);
					return checkForErrors();
			}
		}
		if (is_conv(5, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_input_transform<2, 5, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, xMem.data<T>(),
							tensor_shape, padding, padding_value);
					return checkForErrors();
			}
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	template<typename T, typename U>
	avStatus_t launch_output_transform(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &wDesc, U alpha1,
			const TensorDescriptor &matricesDesc, const MemoryDescriptor &matricesMem, const TensorDescriptor &yDesc, MemoryDescriptor &yMem,
			const TensorDescriptor &bDesc, const MemoryDescriptor &bMem, U alpha2, const TensorDescriptor &zDesc, const MemoryDescriptor &zMem, U beta,
			const avActivationType_t activation, int transformSize)
	{
//			avStatus_t status = TransformSetup::setup();
//			if (status != AVOCADO_STATUS_SUCCESS)
//				return status;
//
//			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(yDesc));
//			MatrixShape<6> matrix_shape = get_matrix_shape < 6 > (cuda::getTensor(matricesDesc));
//
//			int batch_size = cuda::getTensor(yDesc).dimension(0);
//			int tile_h = (cuda::getTensor(yDesc).dimension(1) + 3) / 4;
//			int tile_w = (cuda::getTensor(yDesc).dimension(2) + 3) / 4;
//			int filters_in = cuda::getTensor(yDesc).dimension(3);
//
//			switch (cuda::getTensor(yDesc).dtype())
//			{
//				case AVOCADO_DTYPE_FLOAT32:
//				{
//					dim3 blockDim(128, 4);
//					float _alpha1 = cuda::getAlphaValue(alpha1);
//					float _alpha2 = cuda::getAlphaValue(alpha2);
//					float _beta = cuda::getBetaValue(beta);
//					kernel_winograd_output_transform<4, 3, 128, ActivationLinear<float>> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
//							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(yMem), tensor_shape, cuda::getPointer<float>(zMem),
//							_alpha1, _alpha2, _beta, cuda::getPointer<float>(bMem));
//				}
//					break;
//			}

		context.setDevice();
		TensorShape tensor_shape(yDesc);
		MatrixShape matrix_shape(matricesDesc);

		const int filter_size = wDesc.dimension(1);

		const int batch_size = yDesc.dimension(0);
		const int tile_h = (yDesc.dimension(1) + transformSize - 1) / transformSize;
		const int tile_w = (yDesc.dimension(2) + transformSize - 1) / transformSize;
		const int filters_in = yDesc.dimension(3);

		constexpr int Elements = std::is_same<T, double>::value ? 64 : 128;
		dim3 blockDim(Elements, transformSize);
		dim3 gridDim(batch_size, tile_h, tile_w);
		cudaStream_t stream = context.getStream();

		if (is_conv(3, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_output_transform<2, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, yMem.data<T>(),
							tensor_shape, zMem.data<T>(), alpha1, alpha2, beta, bMem.data<U>(), activation);
					return checkForErrors();
				case 4:
					kernel_winograd_output_transform<4, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, yMem.data<T>(),
							tensor_shape, zMem.data<T>(), alpha1, alpha2, beta, bMem.data<U>(), activation);
					return checkForErrors();
			}
		}
		if (is_conv(5, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_output_transform<2, 5, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, yMem.data<T>(),
							tensor_shape, zMem.data<T>(), alpha1, alpha2, beta, bMem.data<U>(), activation);
					return checkForErrors();
			}
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	template<typename T>
	avStatus_t launch_gradient_transform(const ContextDescriptor &context, const ConvolutionDescriptor &config, const TensorDescriptor &wDesc,
			const TensorDescriptor &dyDesc, const MemoryDescriptor &dyMem, const TensorDescriptor &matricesDesc, MemoryDescriptor &matricesMem,
			int transformSize)
	{
		context.setDevice();
		TensorShape tensor_shape(dyDesc);
		MatrixShape matrix_shape(matricesDesc);

		const int filter_size = wDesc.dimension(1);

		const int batch_size = dyDesc.dimension(0);
		const int tile_h = (dyDesc.dimension(1) + transformSize - 1) / transformSize;
		const int tile_w = (dyDesc.dimension(2) + transformSize - 1) / transformSize;
		const int filters_in = dyDesc.dimension(3);

		constexpr int Elements = std::is_same<T, double>::value ? 64 : 128;
		dim3 blockDim(Elements, std::max(2, transformSize / 2));
		dim3 gridDim(batch_size, tile_h, tile_w);
		cudaStream_t stream = context.getStream();

		if (is_conv(3, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_gradient_transform<2, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dyMem.data<T>(),
							tensor_shape);
					return checkForErrors();
				case 4:
					kernel_winograd_gradient_transform<4, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dyMem.data<T>(),
							tensor_shape);
					return checkForErrors();
			}
		}
		if (is_conv(5, wDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_gradient_transform<2, 5, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dyMem.data<T>(),
							tensor_shape);
					return checkForErrors();
			}
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}

	template<typename T>
	avStatus_t launch_update_transform(const ContextDescriptor &context, const ConvolutionDescriptor &config, T alpha, const TensorDescriptor &matricesDesc,
			const MemoryDescriptor &matricesMem, T beta, const TensorDescriptor &dwDesc, MemoryDescriptor &dwMem, int transformSize)
	{
		context.setDevice();
		TensorShape tensor_shape(dwDesc);
		MatrixShape matrix_shape(matricesDesc);

		const int filter_size = dwDesc.dimension(1);

		const int filters_out = dwDesc.firstDim();
		const int filters_in = dwDesc.lastDim();

		constexpr int Elements = std::is_same<T, double>::value ? 64 : 128;
		dim3 blockDim(Elements, filter_size);
		dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
		cudaStream_t stream = context.getStream();

		if (is_conv(3, dwDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_update_transform<2, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dwMem.data<T>(),
							tensor_shape, alpha, beta);
					return AVOCADO_STATUS_SUCCESS;
				case 4:
					kernel_winograd_update_transform<4, 3, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dwMem.data<T>(),
							tensor_shape, alpha, beta);
					return AVOCADO_STATUS_SUCCESS;
			}
		}
		if (is_conv(5, dwDesc))
		{
			switch (transformSize)
			{
				case 2:
					kernel_winograd_update_transform<2, 5, Elements> <<<gridDim, blockDim, 0, stream>>>(matricesMem.data<T>(), matrix_shape, dwMem.data<T>(),
							tensor_shape, alpha, beta);
					return AVOCADO_STATUS_SUCCESS;
				default:
					return AVOCADO_STATUS_NOT_SUPPORTED;
			}
		}
		return AVOCADO_STATUS_NOT_SUPPORTED;
	}
}

namespace avocado
{
	namespace backend
	{

		avStatus_t cudaWinogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (cuda::getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return launch_weight_transform<float16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getMemory(wMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_BFLOAT16:
					return launch_weight_transform<bfloat16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getMemory(wMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_FLOAT32:
					return launch_weight_transform<float>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getMemory(wMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_FLOAT64:
					return launch_weight_transform<double>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getMemory(wMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
			}
			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

		avStatus_t cudaWinogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc,
				avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (cuda::getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return launch_input_transform<float16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(xDesc), cuda::getMemory(xMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_BFLOAT16:
					return launch_input_transform<bfloat16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(xDesc), cuda::getMemory(xMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_FLOAT32:
					return launch_input_transform<float>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(xDesc), cuda::getMemory(xMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_FLOAT64:
					return launch_input_transform<double>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(xDesc), cuda::getMemory(xMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
			}
			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

		avStatus_t cudaWinogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const void *alpha1, const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem,
				const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem,
				const void *alpha2, const avTensorDescriptor_t zDesc, const avMemoryDescriptor_t zMem, const void *beta, avActivationType_t activation)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (cuda::getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT16:
					return launch_output_transform<float16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getAlphaValue(alpha1), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getTensor(yDesc),
							cuda::getMemory(yMem), cuda::getTensor(bDesc), cuda::getMemory(bMem), cuda::getAlphaValue(alpha2), cuda::getTensor(zDesc),
							cuda::getMemory(zMem), cuda::getBetaValue(beta), activation, transformSize);
				case AVOCADO_DTYPE_BFLOAT16:
					return launch_output_transform<bfloat16>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getAlphaValue(alpha1), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getTensor(yDesc),
							cuda::getMemory(yMem), cuda::getTensor(bDesc), cuda::getMemory(bMem), cuda::getAlphaValue(alpha2), cuda::getTensor(zDesc),
							cuda::getMemory(zMem), cuda::getBetaValue(beta), activation, transformSize);
				case AVOCADO_DTYPE_FLOAT32:
					return launch_output_transform<float>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getAlphaValue(alpha1), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getTensor(yDesc),
							cuda::getMemory(yMem), cuda::getTensor(bDesc), cuda::getMemory(bMem), cuda::getAlphaValue(alpha2), cuda::getTensor(zDesc),
							cuda::getMemory(zMem), cuda::getBetaValue(beta), activation, transformSize);
				case AVOCADO_DTYPE_FLOAT64:
					return launch_output_transform<double>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getAlphaValue<double>(alpha1), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getTensor(yDesc),
							cuda::getMemory(yMem), cuda::getTensor(bDesc), cuda::getMemory(bMem), cuda::getAlphaValue<double>(alpha2), cuda::getTensor(zDesc),
							cuda::getMemory(zMem), cuda::getBetaValue<double>(beta), activation, transformSize);
			}
			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

		avStatus_t cudaWinogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize,
				const avTensorDescriptor_t wDesc, const avTensorDescriptor_t dyDesc, const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc,
				avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (cuda::getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return launch_gradient_transform<float>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(dyDesc), cuda::getMemory(dyMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
				case AVOCADO_DTYPE_FLOAT64:
					return launch_gradient_transform<double>(cuda::getContext(context), cuda::getConvolution(config), cuda::getTensor(wDesc),
							cuda::getTensor(dyDesc), cuda::getMemory(dyMem), cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), transformSize);
			}
			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

		avStatus_t cudaWinogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, int transformSize, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			switch (cuda::getTensor(dwDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					return launch_update_transform<float>(cuda::getContext(context), cuda::getConvolution(config), cuda::getAlphaValue(alpha),
							cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getBetaValue(beta), cuda::getTensor(dwDesc),
							cuda::getMemory(dwMem), transformSize);
				case AVOCADO_DTYPE_FLOAT64:
					return launch_update_transform<double>(cuda::getContext(context), cuda::getConvolution(config), cuda::getAlphaValue<double>(alpha),
							cuda::getTensor(matricesDesc), cuda::getMemory(matricesMem), cuda::getBetaValue<double>(beta), cuda::getTensor(dwDesc),
							cuda::getMemory(dwMem), transformSize);
			}
			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
		}

	} /* namespace backend */
} /* namespace avocado */

/*
 * winograd_nonfused.cu
 *
 *  Created on: Dec 29, 2021
 *      Author: Maciej Kozarzewski
 */

#include <backend_descriptors.hpp>

#include "winograd.hpp"
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

	enum class TransformType
	{
		WEIGHT, INPUT, OUTPUT, GRADIENT, UPDATE
	};

	template<TransformType Type, int TransformSize, int KernelSize>
	__host__ __device__ constexpr int transform_size() noexcept
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
	__host__ __device__ constexpr int transform_offset() noexcept
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
	__host__ __device__ constexpr int total_transform_size() noexcept
	{
		return transform_size<TransformType::WEIGHT, TransformSize, KernelSize>() + transform_size<TransformType::INPUT, TransformSize, KernelSize>()
				+ transform_size<TransformType::OUTPUT, TransformSize, KernelSize>() + transform_size<TransformType::GRADIENT, TransformSize, KernelSize>()
				+ transform_size<TransformType::UPDATE, TransformSize, KernelSize>();
	}

	__constant__ half half_transforms_4x4_3x3[total_transform_size<4, 3>()];
	__constant__ float float_transforms_4x4_3x3[total_transform_size<4, 3>()];
	__constant__ double double_transforms_4x4_3x3[total_transform_size<4, 3>()];

	__constant__ half half_transforms_2x2_3x3[total_transform_size<2, 3>()];
	__constant__ float float_transforms_2x2_3x3[total_transform_size<2, 3>()];
	__constant__ double double_transforms_2x2_3x3[total_transform_size<2, 3>()];

	__constant__ half half_transforms_2x2_5x5[total_transform_size<2, 5>()];
	__constant__ float float_transforms_2x2_5x5[total_transform_size<2, 5>()];
	__constant__ double double_transforms_2x2_5x5[total_transform_size<2, 5>()];

	template<TransformType Type, int TransformSize, int KernelSize, typename T>
	struct TileTransform
	{
	};
	template<TransformType Type>
	struct TileTransform<Type, 4, 3, float>
	{
		__host__ __device__ const float* get() const noexcept
		{
			return float_transforms_4x4_3x3 + transform_offset<Type, 4, 3>();
		}
	};
	template<TransformType Type>
	struct TileTransform<Type, 2, 3, float>
	{
		__host__ __device__ const float* get() const noexcept
		{
			return float_transforms_2x2_3x3 + transform_offset<Type, 2, 3>();
		}
	};
	template<TransformType Type>
	struct TileTransform<Type, 2, 5, float>
	{
		__host__ __device__ const float* get() const noexcept
		{
			return float_transforms_2x2_5x5 + transform_offset<Type, 2, 5>();
		}
	};

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

	template<int TransformSize, int KernelSize, int Elements, typename DataType, typename ComputeType = DataType, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_weight_transform(DataType* matrices, MatrixShape<TileSize> matrix_shape, const DataType* weights, TensorShape weights_shape,
			bool invert)
	{
		SharedStorage<KernelSize, KernelSize, Elements, ComputeType> storage;

		for (int filter = blockIdx.x * blockDim.x + threadIdx.x; filter < weights_shape.filters; filter += gridDim.x * blockDim.x)
		{
			for (int i = threadIdx.y; i < square(KernelSize); i += blockDim.y)
			{
				int row = i / KernelSize;
				int col = i % KernelSize;
				int tensor_offset = weights_shape.offset_at(blockIdx.y, row, col, filter);
				if (invert)
					storage.at(KernelSize - 1 - row, KernelSize - 1 - col, threadIdx.x) = weights[tensor_offset];
				else
					storage.at(row, col, threadIdx.x) = weights[tensor_offset];
			}
			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				TileTransform<TransformType::WEIGHT, TransformSize, KernelSize, ComputeType> transform;
				Tile<KernelSize, ComputeType> computed_row = first_transform<TileSize, KernelSize>(storage.data() + threadIdx.x, Elements, transform.get(),
						row);

				int offset = matrix_shape.offset_at(row, 0, blockIdx.y, filter);
				int stride = matrix_shape.stride_to_next_column();
				for (int col = 0; col < TileSize; col++)
				{
					ComputeType tmp = second_transform<TileSize, KernelSize>(computed_row, transform.get(), col);
					matrices[offset + col * stride] = tmp;
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, int Elements, typename DataType, typename ComputeType = DataType, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_input_transform(DataType* matrices, MatrixShape<TileSize> matrix_shape, const DataType* input, TensorShape input_shape,
			int2 padding)
	{
		SharedStorage<TileSize, TileSize, Elements, ComputeType> storage;

		for (int tile_w = 0; tile_w < input_shape.width; tile_w += TransformSize)
		{
			int filter = blockIdx.x * blockDim.x + threadIdx.x;
			if (filter < input_shape.filters)
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

						if (x >= 0 and x < input_shape.height and y >= 0 and y < input_shape.width)
						{
							int tensor_offset = input_shape.offset_at(batch, x, y, filter);
							storage.at(row, col, threadIdx.x) = input[tensor_offset];
						}
						else
							storage.at(row, col, threadIdx.x) = zero<ComputeType>();
					}
				}
			__syncthreads();

			if (filter < input_shape.filters)
				for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
				{
					TileTransform<TransformType::INPUT, TransformSize, KernelSize, ComputeType> transform;
					Tile<TileSize, ComputeType> computed_row = first_transform<TileSize, TileSize>(storage.data() + threadIdx.x, Elements, transform.get(),
							row);

					int tile_index = input_shape.tile_index<TransformSize>(blockIdx.y, blockIdx.z, tile_w / TransformSize);
					int offset = matrix_shape.offset_at(row, 0, tile_index, filter);
					int stride = matrix_shape.stride_to_next_column();
					for (int col = 0; col < TileSize; col++)
					{
						ComputeType tmp = second_transform<TileSize, TileSize>(computed_row, transform.get(), col);
						matrices[offset + col * stride] = tmp;
					}
				}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, unsigned int Elements, class Activation, typename DataType, typename ComputeType = DataType, int TileSize =
			TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_output_transform(const DataType* matrices, MatrixShape<TileSize> matrix_shape, DataType* output, TensorShape output_shape,
			const DataType* add, ComputeType alpha1, ComputeType alpha2, ComputeType beta, const ComputeType* bias)
	{
		SharedStorage<TileSize, TileSize, Elements, ComputeType> storage;
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
			__syncthreads();

			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
			{
				TileTransform<TransformType::OUTPUT, TransformSize, KernelSize, ComputeType> transform;
				Tile<TileSize, ComputeType> computed_row = first_transform<TransformSize, TileSize>(storage.data() + threadIdx.x, Elements, transform.get(),
						row);

				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;
					if (x < output_shape.height and y < output_shape.width)
					{
						ComputeType tmp = alpha1 * second_transform<TransformSize, TileSize>(computed_row, transform.get(), col);

						if (bias != nullptr)
							tmp += bias[filter];
						int index = output_shape.offset_at(batch, x, y, filter);
						if (add != nullptr)
							tmp += alpha2 * add[index];
						tmp = Activation().forward(tmp);
						if (beta != zero<ComputeType>())
							tmp += beta * output[index];
						output[index] = tmp;
					}
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, int Elements, typename DataType, typename ComputeType = DataType, int TileSize = TransformSize + KernelSize - 1>
	__global__ void kernel_winograd_gradient_transform(DataType* matrices, MatrixShape<TileSize> matrix_shape, const DataType* gradient,
			TensorShape gradient_shape)
	{
		assert(blockDim.x == Elements);
		SharedStorage<TransformSize, TransformSize, Elements, ComputeType> storage;

		for (int filter = threadIdx.x; filter < gradient_shape.filters; filter += Elements)
		{
			for (int row = threadIdx.y; row < TransformSize; row += blockDim.y)
				for (int col = 0; col < TransformSize; col++)
				{
					int batch = blockIdx.x;
					int x = blockIdx.y * TransformSize + row;
					int y = blockIdx.z * TransformSize + col;

					if (x < gradient_shape.height and y < gradient_shape.width)
					{
						int tensor_offset = gradient_shape.offset_at(batch, x, y, filter);
						storage.at(row, col, threadIdx.x) = gradient[tensor_offset];
					}
					else
						storage.at(row, col, threadIdx.x) = zero<ComputeType>();
				}
			__syncthreads();

			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				TileTransform<TransformType::GRADIENT, TransformSize, KernelSize, ComputeType> transform;
				Tile<TransformSize, ComputeType> computed_row = first_transform<TileSize, TransformSize>(storage.data() + threadIdx.x, Elements,
						transform.get(), row);

				int tile_index = gradient_shape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
				int offset = matrix_shape.offset_at(row, 0, tile_index, filter);
				int stride = matrix_shape.stride_to_next_column();
				for (int col = 0; col < TileSize; col++)
				{
					ComputeType tmp = second_transform<TileSize, TransformSize>(computed_row, transform.get(), col);
					matrices[offset + col * stride] = tmp;
				}
			}
			__syncthreads();
		}
	}
	template<int TransformSize, int KernelSize, unsigned int Elements, typename DataType, typename ComputeType = DataType, int TileSize = TransformSize
			+ KernelSize - 1>
	__global__ void kernel_winograd_update_transform(const DataType* matrices, MatrixShape<TileSize> matrix_shape, DataType* update, TensorShape update_shape,
			ComputeType alpha, ComputeType beta)
	{
		SharedStorage<TileSize, TileSize, Elements, ComputeType> storage;
		for (int filter = threadIdx.x; filter < matrix_shape.filters; filter += blockDim.x)
		{
			int tile_index = update_shape.tile_index<TransformSize>(blockIdx.x, blockIdx.y, blockIdx.z);
			for (int row = threadIdx.y; row < TileSize; row += blockDim.y)
			{
				int offset = matrix_shape.offset_at(row, 0, tile_index, filter);
				int stride = matrix_shape.stride_to_next_column();
				for (int col = 0; col < TileSize; col++)
					storage.at(row, col, threadIdx.x) = matrices[offset + col * stride];
			}
			__syncthreads();

			for (int row = threadIdx.y; row < KernelSize; row += blockDim.y)
			{
				TileTransform<TransformType::UPDATE, TransformSize, KernelSize, ComputeType> transform;
				Tile<TileSize, ComputeType> computed_row = first_transform<KernelSize, TileSize>(storage.data() + threadIdx.x, Elements, transform.get(), row);

				for (int col = 0; col < KernelSize; col++)
				{
					ComputeType tmp = alpha * second_transform<KernelSize, TileSize>(computed_row, transform.get(), col);

					int index = update_shape.offset_at(blockIdx.x, row, col, filter);
					if (beta != zero<ComputeType>())
						tmp += beta * update[index];
					update[index] = tmp;
				}
			}
			__syncthreads();
		}
	}

	TensorShape get_tensor_shape(const cuda::TensorDescriptor &desc)
	{
		return TensorShape( { desc.dimension(0), desc.dimension(1), desc.dimension(2), desc.dimension(3) });
	}
	template<int TileSize>
	MatrixShape<TileSize> get_matrix_shape(const cuda::TensorDescriptor &desc)
	{
		return MatrixShape<TileSize>( { desc.dimension(1), desc.dimension(2) });
	}

	template<typename T>
	size_t bytes(const std::vector<T> &a) noexcept
	{
		return sizeof(T) * a.size();
	}
	template<typename T, typename U>
	std::vector<T> cast_to(const std::vector<U> &x)
	{
		std::vector<T> result(x.size());
		for (size_t i = 0; i < x.size(); i++)
			result[i] = static_cast<T>(x[i]);
		return result;
	}

	struct TransformSetup
	{
	private:
		static void setup_4x4_3x3()
		{
			std::vector<double> transform = {
			/* weight */1.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0, -2.0
					/ 3.0, 4.0 / 3.0, 0.0, 0.0, 2.0
			/* input */, 1.0, 0.0, -1.25, 0.0, 0.25, 0.0, 0.0, 1.0, 1.0, -0.25, -0.25, 0.0, 0.0, -1.0, 1.0, 0.25, -0.25, 0.0, 0.0, -1.0, -0.5, 1.0, 0.5, 0.0,
					0.0, 1.0, -0.5, -1.0, 0.5, 0.0, 0.0, 1.0, 0.0, -1.25, 0.0, 0.25
					/* output */, 1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 2.0
					/* gradient */, 1.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0
							/ 3.0, 4.0 / 3.0, 8.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 4.0 / 3.0, -8.0 / 3.0, 0.0, 0.0, 0.0, 2.0
					/*  * update */, 1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0 };

			cudaError_t error1 = cudaMemcpyToSymbol(double_transforms_4x4_3x3, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);
//			CHECK_CUDA_ERROR(error1); FIXME

			std::vector<float> tmp = cast_to<float>(transform);
			cudaError_t error2 = cudaMemcpyToSymbol(float_transforms_4x4_3x3, tmp.data(), bytes(tmp), 0, cudaMemcpyHostToDevice);
//			CHECK_CUDA_ERROR(error2); FIXME

			std::vector<half> tmp2 = cast_to<half>(transform);
			cudaError_t error3 = cudaMemcpyToSymbol(half_transforms_4x4_3x3, tmp2.data(), bytes(tmp2), 0, cudaMemcpyHostToDevice);
//			CHECK_CUDA_ERROR(error3); FIXME
		}
		static void setup_2x2_3x3()
		{
			std::vector<double> transform = {
			/* weight */1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.0, 0.0, 1.0,
			/* input */1.0, 0.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0, -1.0, 0.0, 1.0,
			/* output */1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 1.0,
			/* gradient */1.0, 0.0, 0.5, 0.5, 0.5, -0.5, 0.0, 1.0,
			/* update */1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0 };

			cudaError_t error1 = cudaMemcpyToSymbol(double_transforms_2x2_3x3, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);

			std::vector<float> tmp = cast_to<float>(transform);
			cudaError_t error2 = cudaMemcpyToSymbol(float_transforms_2x2_3x3, tmp.data(), bytes(tmp), 0, cudaMemcpyHostToDevice);

			std::vector<half> tmp2 = cast_to<half>(transform);
			cudaError_t error3 = cudaMemcpyToSymbol(half_transforms_2x2_3x3, tmp2.data(), bytes(tmp2), 0, cudaMemcpyHostToDevice);
		}
		static void setup_2x2_5x5()
		{
			std::vector<double> transform = {
			/* weight */1.0, 0.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 2.0 / 3.0,
					1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, 1.0 / 6.0, -1.0 / 3.0, 2.0 / 3.0, -4.0 / 3.0, 8.0 / 3.0, 0.0, 0.0, 0.0, 0.0, 2.0,
					/* input */1.0, 0.0, -1.25, 0.0, 0.25, 0.0, 0.0, 2.0, 2.0, -0.5, -0.5, 0.0, 0.0, -2.0, 2.0, 0.5, -0.5, 0.0, 0.0, -1.0, -0.5, 1.0, 0.5, 0.0,
					0.0, 1.0, -0.5, -1.0, 0.5, 0.0, 0.0, 1.0, 0.0, -1.25, 0.0, 0.25,
					/* output */1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 2.0,
					/* gradient */1.0, 0.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0, 2.0 / 3.0, 1.0 / 3.0, -2.0 / 3.0, 0.0, 1.0,
					/* update */1.0, 1.0, 1.0, 0.25, 0.25, 0.0, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 0.0,
					0.0, 1.0, 1.0, 4.0, 4.0, 4.0 };

			cudaError_t error1 = cudaMemcpyToSymbol(double_transforms_2x2_5x5, transform.data(), bytes(transform), 0, cudaMemcpyHostToDevice);

			std::vector<float> tmp = cast_to<float>(transform);
			cudaError_t error2 = cudaMemcpyToSymbol(float_transforms_2x2_5x5, tmp.data(), bytes(tmp), 0, cudaMemcpyHostToDevice);

			std::vector<half> tmp2 = cast_to<half>(transform);
			cudaError_t error3 = cudaMemcpyToSymbol(half_transforms_2x2_5x5, tmp2.data(), bytes(tmp2), 0, cudaMemcpyHostToDevice);
		}
	public:
		static avStatus_t setup()
		{
			static bool already_done = []()
			{
				try
				{
					setup_4x4_3x3();
					setup_2x2_3x3();
					setup_2x2_5x5();
					return true;
				}
				catch(std::exception &e)
				{
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
	avStatus_t launcher_weight_transform(const cuda::ContextDescriptor& context, const cuda::ConvolutionDescriptor& config, const cuda::TensorDescriptor& wDesc,
			const T* wMem, const cuda::TensorDescriptor& matricesDesc, T* matricesMem)
	{
		TensorShape tensor_shape = get_tensor_shape(wDesc);
		MatrixShape<6> matrix_shape = get_matrix_shape<6>(matricesDesc);

		int filters_out = wDesc.firstDim();
		int filters_in = wDesc.lastDim();
		dim3 blockDim(128, 1);
		dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
		cudaStream_t stream = context.getStream();

		bool invert = (config.mode == AVOCADO_CROSS_CORRELATION_MODE);

		switch (wDesc.dtype())
		{
			case AVOCADO_DTYPE_FLOAT32:
				kernel_winograd_weight_transform<4, 3, 128> <<<gridDim, blockDim, sizeof(float) * 3 * 3 * blockDim.x, stream>>>(matricesMem, matrix_shape, wMem,
						tensor_shape, invert);
				break;
		}
		return AVOCADO_STATUS_SUCCESS;
	}
}

namespace avocado
{
	namespace backend
	{

		avSize_t cuda_winogradGetWorkspaceSize(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avTensorDescriptor_t wDesc)
		{
			if (cuda::getConvolution(config).algorithm == AVOCADO_CONVOLUTION_ALGORITHM_EXPLICIT_GEMM)
			{
				return 0; // TODO
			}
			if (cuda::getConvolution(config).algorithm == AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_NON_FUSED)
			{
				switch (cuda::getTensor(wDesc).dtype())
				{
					case AVOCADO_DTYPE_INT8:
						return 0; // TODO
					case AVOCADO_DTYPE_FLOAT16:
					case AVOCADO_DTYPE_BFLOAT16:
					case AVOCADO_DTYPE_FLOAT32:
					case AVOCADO_DTYPE_FLOAT64:
						return 0; // TODO
				}
			}
			if (cuda::getConvolution(config).algorithm == AVOCADO_CONVOLUTION_ALGORITHM_WINOGRAD_FUSED)
			{
				if (cuda::getTensor(wDesc).dtype() == AVOCADO_DTYPE_INT8)
					return 0; // TODO
			}
			return 0;
		}

		avStatus_t cuda_winogradWeightTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t wDesc,
				const avMemoryDescriptor_t wMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(wDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(cuda::getTensor(matricesDesc));

			int filters_out = cuda::getTensor(wDesc).firstDim();
			int filters_in = cuda::getTensor(wDesc).lastDim();
			dim3 blockDim(128, 1);
			dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(wDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_winograd_weight_transform<4, 3, 128> <<<gridDim, blockDim, sizeof(float) * 3 * 3 * blockDim.x, stream>>>(
							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(wMem), tensor_shape, false);
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cuda_winogradInputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t xDesc,
				const avMemoryDescriptor_t xMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(xDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(cuda::getTensor(matricesDesc));

			int batch_size = cuda::getTensor(xDesc).dimension(0);
			int tile_h = (cuda::getTensor(xDesc).dimension(1) + 3) / 4;
			int tile_w = (cuda::getTensor(xDesc).dimension(2) + 3) / 4;
			int filters_in = cuda::getTensor(xDesc).dimension(3);

			int2 padding { -1, -1 };

			dim3 blockDim(128, 3);
			dim3 gridDim((filters_in + blockDim.x - 1) / blockDim.x, batch_size, tile_h);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(xDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
					kernel_winograd_input_transform<4, 3, 128> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(xMem), tensor_shape, padding);
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cuda_winogradOutputTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avActivationType_t activation)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(yDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(cuda::getTensor(matricesDesc));

			int batch_size = cuda::getTensor(yDesc).dimension(0);
			int tile_h = (cuda::getTensor(yDesc).dimension(1) + 3) / 4;
			int tile_w = (cuda::getTensor(yDesc).dimension(2) + 3) / 4;
			int filters_in = cuda::getTensor(yDesc).dimension(3);

			dim3 gridDim(batch_size, tile_h, tile_w);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					dim3 blockDim(128, 4);
					float _alpha1 = cuda::getAlphaValue(alpha1);
					float _alpha2 = cuda::getAlphaValue(alpha2);
					float _beta = cuda::getBetaValue(beta);
					kernel_winograd_output_transform<4, 3, 128, ActivationLinear<float>> <<<gridDim, blockDim, 6 * 6 * blockDim.x * sizeof(float), stream>>>(
							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(yMem), tensor_shape, cuda::getPointer<float>(zMem),
							_alpha1, _alpha2, _beta, cuda::getPointer<float>(bMem));
				}
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cuda_winogradGradientTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const avTensorDescriptor_t dyDesc,
				const avMemoryDescriptor_t dyMem, const avTensorDescriptor_t matricesDesc, avMemoryDescriptor_t matricesMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(dyDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(cuda::getTensor(matricesDesc));

			int batch_size = cuda::getTensor(dyDesc).dimension(0);
			int tile_h = (cuda::getTensor(dyDesc).dimension(1) + 3) / 4;
			int tile_w = (cuda::getTensor(dyDesc).dimension(2) + 3) / 4;
			int filters_in = cuda::getTensor(dyDesc).dimension(3);

			dim3 gridDim(batch_size, tile_h, tile_w);
			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(dyDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					dim3 blockDim(128, 2);
					kernel_winograd_gradient_transform<4, 3, 128> <<<gridDim, blockDim, 4 * 4 * blockDim.x * sizeof(float), stream>>>(
							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(dyMem), tensor_shape);
					break;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}

		avStatus_t cuda_winogradUpdateTransform(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha,
				const avTensorDescriptor_t matricesDesc, const avMemoryDescriptor_t matricesMem, const void *beta, const avTensorDescriptor_t dwDesc,
				avMemoryDescriptor_t dwMem)
		{
			avStatus_t status = TransformSetup::setup();
			if (status != AVOCADO_STATUS_SUCCESS)
				return status;

			TensorShape tensor_shape = get_tensor_shape(cuda::getTensor(dwDesc));
			MatrixShape<6> matrix_shape = get_matrix_shape<6>(cuda::getTensor(matricesDesc));

			int filters_out = cuda::getTensor(dwDesc).firstDim();
			int filters_in = cuda::getTensor(dwDesc).lastDim();

			cudaStream_t stream = cuda::getContext(context).getStream();
			cuda::getContext(context).setDevice();

			switch (cuda::getTensor(dwDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					dim3 blockDim(128, 3);
					dim3 gridDim(gridSize<32>(filters_in, blockDim.x), filters_out);
					float _alpha = cuda::getAlphaValue(alpha);
					float _beta = cuda::getBetaValue(beta);
					kernel_winograd_update_transform<4, 3, 128> <<<gridDim, blockDim, sizeof(float) * 6 * 6 * blockDim.x, stream>>>(
							cuda::getPointer<float>(matricesMem), matrix_shape, cuda::getPointer<float>(dwMem), tensor_shape, _alpha, _beta);
					break;
				}
			}
			return AVOCADO_STATUS_SUCCESS;
		}

	} /* namespace backend */
} /* namespace avocado */

/*
 * winograd_nonfused.cu
 *
 *  Created on: Dec 29, 2021
 *      Author: Maciej Kozarzewski
 */

#include <Avocado/cuda_backend.h>
#include <Avocado/backend_descriptors.hpp>

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
	using namespace avocado::backend::BACKEND_NAMESPACE;

	template<typename T, int Rows, int Columns = Rows>
	struct Tile
	{
	private:
		T data[Rows * Columns];
	public:
		__device__ Tile()
		{
		}
		__device__ Tile(T value)
		{
			for (int i = 0; i < Rows * Columns; i++)
				data[i] = value;
		}
		__device__ void print() const noexcept
		{
			for (int r = 0; r < Rows; r++)
			{
				for (int c = 0; c < Columns; c++)
					printf("%f ", data[r * Columns + c]);
				printf("\n");
			}
			printf("\n");
		}
		__device__ Tile<T, Rows, Columns>& operator=(T value) noexcept
		{
			for (int i = 0; i < Rows * Columns; i++)
				data[i] = value;
			return *this;
		}
		__device__ T operator[](int index) const noexcept
		{
			assert(index >= 0 && index < Rows * Columns);
			return data[index];
		}
		__device__ T& operator[](int index) noexcept
		{
			assert(index >= 0 && index < Rows * Columns);
			return data[index];
		}
		__device__ T at(int row, int col) const noexcept
		{
			assert(row >= 0 && row < Rows);
			assert(col >= 0 && col < Columns);
			return data[row * Columns + col];
		}
		__device__ T& at(int row, int col) noexcept
		{
			assert(row >= 0 && row < Rows);
			assert(col >= 0 && col < Columns);
			return data[row * Columns + col];
		}
		__device__ void fma(const Tile<T, Rows, Columns> &lhs, const Tile<T, Rows, Columns> &rhs) noexcept
		{
			for (int i = 0; i < Rows * Columns; i += 2)
			{
				data[i] += lhs.data[i] * rhs.data[i];
				data[i + 1] += lhs.data[i + 1] * rhs.data[i + 1];
			}
		}
		__device__ void load(const T* src, const int stride) noexcept
		{
			int tmp = 0;
			for (int i = 0; i < Rows * Columns; i++, tmp += stride)
				data[i] = src[tmp];
		}
		__device__ void store(T* dst, const int stride) const noexcept
		{
			int tmp = 0;
			for (int i = 0; i < Rows * Columns; i++, tmp += stride)
				dst[tmp] = data[i];
		}
	};

	template<typename T>
	struct Tile4x4
	{
		T x00, x01, x02, x03;
		T x10, x11, x12, x13;
		T x20, x21, x22, x23;
		T x30, x31, x32, x33;

		__device__ Tile4x4()
		{
		}
		__device__ Tile4x4(T value)
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

		__device__ void load_and_transform(const T* src, const int row_stride, const int col_stride) noexcept
		{
#define get(x, y) src[(x) * row_stride + (y) * col_stride]

//			x00 = src[0 * row_stride + 0 * col_stride];
//			x01 = src[0 * row_stride + 1 * col_stride];
//			x02 = src[0 * row_stride + 2 * col_stride];
//			x03 = src[0 * row_stride + 3 * col_stride];
//
//			x10 = src[1 * row_stride + 0 * col_stride];
//			x11 = src[1 * row_stride + 1 * col_stride];
//			x12 = src[1 * row_stride + 2 * col_stride];
//			x13 = src[1 * row_stride + 3 * col_stride];
//
//			x20 = src[2 * row_stride + 0 * col_stride];
//			x21 = src[2 * row_stride + 1 * col_stride];
//			x22 = src[2 * row_stride + 2 * col_stride];
//			x23 = src[2 * row_stride + 3 * col_stride];
//
//			x30 = src[3 * row_stride + 0 * col_stride];
//			x31 = src[3 * row_stride + 1 * col_stride];
//			x32 = src[3 * row_stride + 2 * col_stride];
//			x33 = src[3 * row_stride + 3 * col_stride];
//
//			T tmp00 = x00 - x20;
//			T tmp01 = x01 - x21;
//			T tmp02 = x02 - x22;
//			T tmp03 = x03 - x23;
//
//			T tmp10 = x10 + x20;
//			T tmp11 = x11 + x21;
//			T tmp12 = x12 + x22;
//			T tmp13 = x13 + x23;
//
//			T tmp20 = -x10 + x20;
//			T tmp21 = -x11 + x21;
//			T tmp22 = -x12 + x22;
//			T tmp23 = -x13 + x23;
//
//			T tmp30 = -x10 + x30;
//			T tmp31 = -x11 + x31;
//			T tmp32 = -x12 + x32;
//			T tmp33 = -x13 + x33;

			T tmp00 = get(0,0)- get(2,0);
			T tmp01 = get(0,1)- get(2,1);
			T tmp02 = get(0,2)- get(2,2);
			T tmp03 = get(0,3)- get(2,3);

			T tmp10 = get(1,0) + get(2,0);
			T tmp11 = get(1,1) + get(2,1);
			T tmp12 = get(1,2) + get(2,2);
			T tmp13 = get(1,3) + get(2,3);

			T tmp20 = -get(1,0) + get(2,0);
			T tmp21 = -get(1,0) + get(2,0);
			T tmp22 = -get(1,0) + get(2,0);
			T tmp23 = -get(1,0) + get(2,0);

			T tmp30 = -get(1,0) + get(3,0);
			T tmp31 = -get(1,0) + get(3,0);
			T tmp32 = -get(1,0) + get(3,0);
			T tmp33 = -get(1,0) + get(3,0);

			x00 = tmp00 - tmp02;
			x10 = tmp10 - tmp12;
			x20 = tmp20 - tmp22;
			x30 = tmp30 - tmp32;

			x01 = tmp01 + tmp02;
			x11 = tmp11 + tmp12;
			x21 = tmp21 + tmp22;
			x31 = tmp31 + tmp32;

			x02 = -tmp01 + tmp02;
			x12 = -tmp11 + tmp12;
			x22 = -tmp21 + tmp22;
			x32 = -tmp31 + tmp32;

			x03 = -tmp01 + tmp03;
			x13 = -tmp11 + tmp13;
			x23 = -tmp21 + tmp23;
			x33 = -tmp31 + tmp33;
		}

		__device__ void transform() noexcept
		{
			T tmp00 = x00 - x20;
			T tmp01 = x01 - x21;
			T tmp02 = x02 - x22;
			T tmp03 = x03 - x23;

			T tmp10 = x10 + x20;
			T tmp11 = x11 + x21;
			T tmp12 = x12 + x22;
			T tmp13 = x13 + x23;

			T tmp20 = -x10 + x20;
			T tmp21 = -x11 + x21;
			T tmp22 = -x12 + x22;
			T tmp23 = -x13 + x23;

			T tmp30 = -x10 + x30;
			T tmp31 = -x11 + x31;
			T tmp32 = -x12 + x32;
			T tmp33 = -x13 + x33;

			x00 = tmp00 - tmp02;
			x10 = tmp10 - tmp12;
			x20 = tmp20 - tmp22;
			x30 = tmp30 - tmp32;

			x01 = tmp01 + tmp02;
			x11 = tmp11 + tmp12;
			x21 = tmp21 + tmp22;
			x31 = tmp31 + tmp32;

			x02 = -tmp01 + tmp02;
			x12 = -tmp11 + tmp12;
			x22 = -tmp21 + tmp22;
			x32 = -tmp31 + tmp32;

			x03 = -tmp01 + tmp03;
			x13 = -tmp11 + tmp13;
			x23 = -tmp21 + tmp23;
			x33 = -tmp31 + tmp33;
		}
		__device__ void fma(const Tile4x4 &lhs, const Tile4x4 &rhs) noexcept
		{
			x00 = __fmaf_rn(lhs.x00 , rhs.x00, x00);
			x01 = __fmaf_rn(lhs.x01 , rhs.x01, x01);
			x02 = __fmaf_rn(lhs.x02 , rhs.x02, x02);
			x03 = __fmaf_rn(lhs.x03 , rhs.x03, x03);

			x10 = __fmaf_rn(lhs.x10 , rhs.x10, x10);
			x11 = __fmaf_rn(lhs.x11 , rhs.x11, x11);
			x12 = __fmaf_rn(lhs.x12 , rhs.x12, x12);
			x13 = __fmaf_rn(lhs.x13 , rhs.x13, x13);

			x20 = __fmaf_rn(lhs.x20 , rhs.x20, x20);
			x21 = __fmaf_rn(lhs.x21 , rhs.x21, x21);
			x22 = __fmaf_rn(lhs.x22 , rhs.x22, x22);
			x23 = __fmaf_rn(lhs.x23 , rhs.x23, x23);

			x30 = __fmaf_rn(lhs.x30 , rhs.x30, x30);
			x31 = __fmaf_rn(lhs.x31 , rhs.x31, x31);
			x32 = __fmaf_rn(lhs.x32 , rhs.x32, x32);
			x33 = __fmaf_rn(lhs.x33 , rhs.x33, x33);

//			x00 += lhs.x00 * rhs.x00;
//			x01 += lhs.x01 * rhs.x01;
//			x02 += lhs.x02 * rhs.x02;
//			x03 += lhs.x03 * rhs.x03;
//
//			x10 += lhs.x10 * rhs.x10;
//			x11 += lhs.x11 * rhs.x11;
//			x12 += lhs.x12 * rhs.x12;
//			x13 += lhs.x13 * rhs.x13;
//
//			x20 += lhs.x20 * rhs.x20;
//			x21 += lhs.x21 * rhs.x21;
//			x22 += lhs.x22 * rhs.x22;
//			x23 += lhs.x23 * rhs.x23;
//
//			x30 += lhs.x30 * rhs.x30;
//			x31 += lhs.x31 * rhs.x31;
//			x32 += lhs.x32 * rhs.x32;
//			x33 += lhs.x33 * rhs.x33;
		}
		__device__ void load(const T* src, const int row_stride, const int col_stride) noexcept
		{
			x00 = src[0 * row_stride + 0 * col_stride];
			x01 = src[0 * row_stride + 1 * col_stride];
			x02 = src[0 * row_stride + 2 * col_stride];
			x03 = src[0 * row_stride + 3 * col_stride];

			x10 = src[1 * row_stride + 0 * col_stride];
			x11 = src[1 * row_stride + 1 * col_stride];
			x12 = src[1 * row_stride + 2 * col_stride];
			x13 = src[1 * row_stride + 3 * col_stride];

			x20 = src[2 * row_stride + 0 * col_stride];
			x21 = src[2 * row_stride + 1 * col_stride];
			x22 = src[2 * row_stride + 2 * col_stride];
			x23 = src[2 * row_stride + 3 * col_stride];

			x30 = src[3 * row_stride + 0 * col_stride];
			x31 = src[3 * row_stride + 1 * col_stride];
			x32 = src[3 * row_stride + 2 * col_stride];
			x33 = src[3 * row_stride + 3 * col_stride];
		}
		__device__ void print() const
		{
			printf("%f %f %f %f\n", x00, x01, x02, x03);
			printf("%f %f %f %f\n", x10, x11, x12, x13);
			printf("%f %f %f %f\n", x20, x21, x22, x23);
			printf("%f %f %f %f\n", x30, x31, x32, x33);
			printf("\n");
		}
	};

	template<typename T>
	struct Tile3x3
	{
		T x00, x01, x02;
		T x10, x11, x12;
		T x20, x21, x22;

		__device__ Tile4x4<T> load_and_transform(const T *src, const int stride) noexcept
		{
			x00 = src[0 + 0 * stride];
			x01 = src[0 + 1 * stride];
			x02 = src[0 + 2 * stride];

			x10 = src[0 + 3 * stride];
			x11 = src[0 + 4 * stride];
			x12 = src[0 + 5 * stride];

			x20 = src[0 + 6 * stride];
			x21 = src[0 + 7 * stride];
			x22 = src[0 + 8 * stride];

			T tmp00 = x00;
			T tmp10 = x00 + x10 + x20;
			T tmp20 = x00 - x10 + x20;
			T tmp30 = x20;

			T tmp01 = x01;
			T tmp11 = x01 + x11 + x21;
			T tmp21 = x01 - x11 + x21;
			T tmp31 = x21;

			T tmp02 = x02;
			T tmp12 = x02 + x12 + x22;
			T tmp22 = x02 - x12 + x22;
			T tmp32 = x22;

			Tile4x4<T> result;
			result.x00 = tmp00;
			result.x10 = tmp10;
			result.x20 = tmp20;
			result.x30 = tmp30;

			result.x01 = tmp10 + tmp11 + tmp12;
			result.x11 = tmp10 + tmp11 + tmp12;
			result.x21 = tmp10 + tmp11 + tmp12;
			result.x31 = tmp10 + tmp11 + tmp12;

			result.x02 = tmp20 - tmp21 + tmp22;
			result.x12 = tmp20 - tmp21 + tmp22;
			result.x22 = tmp20 - tmp21 + tmp22;
			result.x32 = tmp20 - tmp21 + tmp22;

			result.x03 = tmp02;
			result.x13 = tmp12;
			result.x23 = tmp22;
			result.x33 = tmp32;
			return result;
		}
		__device__ Tile4x4<T> transform() noexcept
		{
			T tmp00 = x00;
			T tmp10 = x00 + x10 + x20;
			T tmp20 = x00 - x10 + x20;
			T tmp30 = x20;

			T tmp01 = x01;
			T tmp11 = x01 + x11 + x21;
			T tmp21 = x01 - x11 + x21;
			T tmp31 = x21;

			T tmp02 = x02;
			T tmp12 = x02 + x12 + x22;
			T tmp22 = x02 - x12 + x22;
			T tmp32 = x22;

			Tile4x4<T> result;
			result.x00 = tmp00;
			result.x10 = tmp10;
			result.x20 = tmp20;
			result.x30 = tmp30;

			result.x01 = tmp10 + tmp11 + tmp12;
			result.x11 = tmp10 + tmp11 + tmp12;
			result.x21 = tmp10 + tmp11 + tmp12;
			result.x31 = tmp10 + tmp11 + tmp12;

			result.x02 = tmp20 - tmp21 + tmp22;
			result.x12 = tmp20 - tmp21 + tmp22;
			result.x22 = tmp20 - tmp21 + tmp22;
			result.x32 = tmp20 - tmp21 + tmp22;

			result.x03 = tmp02;
			result.x13 = tmp12;
			result.x23 = tmp22;
			result.x33 = tmp32;
			return result;
		}
		__device__ void load(const T* src, const int stride) noexcept
		{
			x00 = src[0 + 0 * stride];
			x01 = src[0 + 1 * stride];
			x02 = src[0 + 2 * stride];

			x10 = src[0 + 3 * stride];
			x11 = src[0 + 4 * stride];
			x12 = src[0 + 5 * stride];

			x20 = src[0 + 6 * stride];
			x21 = src[0 + 7 * stride];
			x22 = src[0 + 8 * stride];
		}
		__device__ void print() const
		{
			printf("%f %f %f\n", x00, x01, x02);
			printf("%f %f %f\n", x10, x11, x12);
			printf("%f %f %f\n", x20, x21, x22);
			printf("\n");
		}
	};

	template<typename T>
	struct Tile2x2
	{
		T x00, x01;
		T x10, x11;

		__device__ Tile2x2(T value = scalar_zero<T>())
		{
			x00 = value;
			x01 = value;
			x10 = value;
			x11 = value;
		}
		__device__ void transform_add(const Tile4x4<T> & tile) noexcept
		{
			T tmp00 = tile.x00 + static_cast<T>(0.5f) * (tile.x10 + tile.x20);
			T tmp01 = tile.x01 + static_cast<T>(0.5f) * (tile.x11 + tile.x21);
			T tmp02 = tile.x02 + static_cast<T>(0.5f) * (tile.x12 + tile.x22);
			T tmp03 = tile.x03 + static_cast<T>(0.5f) * (tile.x13 + tile.x23);

			T tmp10 = static_cast<T>(0.5f) * (tile.x10 - tile.x20) + tile.x30;
			T tmp11 = static_cast<T>(0.5f) * (tile.x11 - tile.x21) + tile.x31;
			T tmp12 = static_cast<T>(0.5f) * (tile.x12 - tile.x22) + tile.x32;
			T tmp13 = static_cast<T>(0.5f) * (tile.x13 - tile.x23) + tile.x33;

			x00 += tmp00 + static_cast<T>(0.5f) * (tmp01 + tmp02);
			x10 += static_cast<T>(0.5f) * (tmp01 - tmp02) + tmp03;
			x01 += tmp10 + static_cast<T>(0.5f) * (tmp11 + tmp12);
			x11 += static_cast<T>(0.5f) * (tmp11 - tmp12) + tmp13;
		}
		__device__ T at(int row, int col) const
		{
			if (row == 0)
			{
				if (col == 0)
					return x00;
				else
					return x01;
			}
			else
			{
				if (col == 0)
					return x10;
				else
					return x11;
			}
		}
		__device__ void print() const
		{
			printf("%f %f\n", x00, x01);
			printf("%f %f\n", x10, x11);
			printf("\n");
		}
	};

	template<typename T>
	__device__ Tile<T, 4> transform_input(const Tile<T, 4> &tile) noexcept
	{
		Tile<T, 4> result;
		for (int i = 0; i < 4; i++)
		{
			result[i] = tile[0 * 4 + i] - tile[2 * 4 + i];
			result[4 + i] = tile[1 * 4 + i] + tile[2 * 4 + i];
			result[2 * 4 + i] = -tile[1 * 4 + i] + tile[2 * 4 + i];
			result[3 * 4 + i] = -tile[1 * 4 + i] + tile[3 * 4 + i];
		}
		for (int i = 0; i < 4; i++)
		{
			T tmp0 = result[i * 4 + 0];
			T tmp1 = result[i * 4 + 1];
			T tmp2 = result[i * 4 + 2];
			T tmp3 = result[i * 4 + 3];
			result[i * 4 + 0] = tmp0 - tmp2;
			result[i * 4 + 1] = tmp1 + tmp2;
			result[i * 4 + 2] = -tmp1 + tmp2;
			result[i * 4 + 3] = -tmp1 + tmp3;
		}
		return result;
	}

	template<typename T>
	__device__ Tile<T, 4> transform_weight(const Tile<T, 3> &tile) noexcept
	{
		Tile<T, 4> result;
		for (int i = 0; i < 3; i++)
		{
			result[0 * 4 + i] = tile[0 * 3 + i];
			result[1 * 4 + i] = tile[0 * 3 + i] + tile[1 * 3 + i] + tile[2 * 3 + i];
			result[2 * 4 + i] = tile[0 * 3 + i] - tile[1 * 3 + i] + tile[2 * 3 + i];
			result[3 * 4 + i] = tile[3 * 3 + i];
		}
		for (int i = 0; i < 4; i++)
		{
			T tmp0 = result[i * 4 + 0];
			T tmp1 = result[i * 4 + 1];
			T tmp2 = result[i * 4 + 2];
			result[i * 4 + 0] = tmp0;
			result[i * 4 + 1] = tmp0 + tmp1 + tmp2;
			result[i * 4 + 2] = tmp0 - tmp1 + tmp2;
			result[i * 4 + 3] = tmp2;
		}
		return result;
	}

	template<typename T>
	__device__ Tile<T, 2> transform_output(const Tile<T, 4> &tile) noexcept
	{
		Tile<T, 2, 4> tmp;
		for (int i = 0; i < 4; i++)
		{
			tmp[0 * 4 + i] = tile[0 * 4 + i] + static_cast<T>(0.5f) * (tile[1 * 4 + i] + tile[2 * 4 + i]);
			tmp[1 * 4 + i] = static_cast<T>(0.5f) * (tile[1 * 4 + i] - tile[2 * 4 + i]) + tile[3 * 4 + i];
		}
		Tile<T, 2> result;
		for (int i = 0; i < 2; i++)
		{
			result[2 * i + 0] = tmp[4 * i + 0] + static_cast<T>(0.5f) * (tmp[4 * i + 1] + tmp[4 * i + 2]);
			result[2 * i + 1] = static_cast<T>(0.5f) * (tmp[4 * i + 1] - tmp[4 * i + 2]) + tmp[4 * i + 3];
		}
		return result;
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

	template<typename T, class Activation, int KernelSize = 3, int TransformSize = 2, int OutputTileSize = 8, int OutputFilterFragments = 64,
			int InputFilterFragments = 8>
	__launch_bounds__(256, 2)
	__global__ void kernel_winograd_fused(const T* weights, const T* input, TensorShape input_shape, T* output, TensorShape output_shape, const int2 padding,
			T alpha1, T alpha2, T beta, const T* bias = nullptr, const T* add = nullptr)
	{
		assert(blockDim.x == 256);
		constexpr int InputTileSize = OutputTileSize + KernelSize - 1;

		__shared__ T input_storage[square(OutputTileSize + KernelSize - 1) * InputFilterFragments];
		__shared__ T input_tiles[square(TransformSize + KernelSize - 1) * square(OutputTileSize / 4)];
		__shared__ T weight_storage[OutputFilterFragments * KernelSize * KernelSize * InputFilterFragments];
		__shared__ T weight_tiles[square(TransformSize + KernelSize - 1) * OutputFilterFragments];

		/* Four 2x2 accumulators represent following output data layout
		 *
		 *  For threadIdx.x == 0
		 *  acc00.at(0, 0), acc00.at(0, 1) ... 1 other accumulators ... acc01.at(0, 0), acc01.at(0, 1) ... 3 other accumulators ... END
		 *  acc00.at(1, 0), acc00.at(1, 1) ... 1 other accumulators ... acc01.at(1, 0), acc01.at(1, 1) ... 3 other accumulators ... END
		 *  ... 2 other lines ...
		 *  acc10.at(1, 0), acc10.at(1, 1) ... 3 other accumulators ... acc11.at(1, 0), acc11.at(1, 1) ... 3 other accumulators ... END
		 *  acc10.at(1, 0), acc10.at(1, 1) ... 3 other accumulators ... acc11.at(1, 0), acc11.at(1, 1) ... 3 other accumulators ... END
		 *  ... 6 other lines ...
		 */
		Tile2x2<T> acc00;
		Tile2x2<T> acc01;
		Tile2x2<T> acc10;
		Tile2x2<T> acc11;
		for (int input_filter = 0; input_filter < input_shape.filters; input_filter += InputFilterFragments)
		{
			int2 tmp_thread_idx = split_thread_index(InputFilterFragments); // divide into 8x32
			int2 tmp_block_dim = split_block_dim(InputFilterFragments); // divide into 8x32

			int tmp = (output_shape.filters + OutputFilterFragments - 1) / OutputFilterFragments;
			int batch = blockIdx.x / tmp;

			// input_storage [InputTileSize x InputTileSize x InputFilterFragments]
			for (int i = tmp_thread_idx.y; i < square(OutputTileSize + KernelSize - 1); i += tmp_block_dim.y)
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
					input_storage[index] = input[offset];
				}
				else
					input_storage[index] = scalar_zero<T>();
			}

			// weight_storage [OutputFilterFragments x KernelSize x KernelSize x InputFilterFragments]
			int output_filter = (blockIdx.x % tmp) * OutputFilterFragments;
			for (int out = tmp_thread_idx.y; out < OutputFilterFragments; out += tmp_block_dim.y)
				for (int i = 0; i < square(KernelSize); i++)
				{
					int offset = ((output_filter + out) * square(KernelSize) + i) * input_shape.filters + input_filter + tmp_thread_idx.x;
					int index = (out * square(KernelSize) + i) * InputFilterFragments + tmp_thread_idx.x;
					weight_storage[index] = weights[offset];
				}
//			__syncthreads();
//			if (blockIdx.y == 1 and blockIdx.z == 1 and threadIdx.x == 0 and input_filter == 0)
//			{
//				for (int i = 0; i < 10; i++)
//				{
//					for (int j = 0; j < 10; j++)
//						printf("%f ", input_storage[(i * 10 + j) * InputFilterFragments + 7]);
//					printf("\n");
//				}
//			}
//			__syncthreads();
//			if (blockIdx.y == 1 and blockIdx.z == 1 and threadIdx.x == 0 and input_filter == 0)
//			{
//				for (int i = 0; i < 3; i++)
//				{
//					for (int j = 0; j < 3; j++)
//						printf("%f ", weight_storage[(23 * 9 + i * 3 + j) * InputFilterFragments + 4]);
//					printf("\n");
//				}
//			}

			__syncthreads();

			Tile4x4<T> tile00(scalar_zero<T>());
			Tile4x4<T> tile01(scalar_zero<T>());
			Tile4x4<T> tile10(scalar_zero<T>());
			Tile4x4<T> tile11(scalar_zero<T>());
			for (int k = 0; k < InputFilterFragments; k++)
			{
				//here transform input and weight data into tiles

				//here load tiles into registers
				Tile4x4<T> weight_tile0(1.0f);
				Tile4x4<T> weight_tile1(1.0f);

				int2 tmp_thread_idx = split_thread_index(InputFilterFragments); // divide into 8x32

				Tile4x4<T> input_tile0;
				input_tile0.load_and_transform(input_storage + 0, InputTileSize * InputFilterFragments, InputFilterFragments);
				Tile4x4<T> input_tile1;
				input_tile1.load_and_transform(input_storage + 0, InputTileSize * InputFilterFragments, InputFilterFragments);

//				if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and input_filter == 0)
//				{
//					printf("loaded\n");
//					input_tile0.print();
//					input_tile1.print();
//				}
//				input_tile0.transform();
//				input_tile1.transform();
//				if (blockIdx.y == 0 and blockIdx.z == 0 and threadIdx.x == 0 and input_filter == 0)
//				{
//					printf("transformed\n");
//					input_tile0.print();
//					input_tile1.print();
//				}
				tile00.fma(input_tile0, weight_tile0);
				tile01.fma(input_tile0, weight_tile1);
				tile10.fma(input_tile1, weight_tile0);
				tile11.fma(input_tile1, weight_tile1);
			}
			acc00.transform_add(tile00);
			acc01.transform_add(tile01);
			acc10.transform_add(tile10);
			acc11.transform_add(tile11);
			//here perform output transform
			__syncthreads();
		}
//		if (threadIdx.x == 0 and threadIdx.y == 0)
//		{
//			tile00.print();
//			tile01.print();
//			tile10.print();
//			tile11.print();
//		}
//		__syncthreads();

//		Tile<T, 2> acc00 = transform_output(tile00);
//		Tile<T, 2> acc01 = transform_output(tile01);
//		Tile<T, 2> acc10 = transform_output(tile10);
//		Tile<T, 2> acc11 = transform_output(tile11);

		int2 tmp_thread_idx = split_thread_index(OutputFilterFragments); // divide into 64x4
		int2 tmp_block_dim = split_block_dim(OutputFilterFragments); // divide into 64x4
		for (int i = 0; i < 2; i++)
			for (int j = 0; j < 2; j++)
			{
				int x = 2 * (tmp_thread_idx.y / 2) + i;
				int y = 2 * (tmp_thread_idx.y % 2) + j;
				int tmp = OutputTileSize / 2;
				weight_storage[(x * OutputTileSize + y) * OutputFilterFragments + tmp_thread_idx.x] = acc00.at(i, j);
				weight_storage[(x * OutputTileSize + y + tmp) * OutputFilterFragments + tmp_thread_idx.x] = acc01.at(i, j);
				weight_storage[((x + tmp) * OutputTileSize + y) * OutputFilterFragments + tmp_thread_idx.x] = acc10.at(i, j);
				weight_storage[((x + tmp) * OutputTileSize + y + tmp) * OutputFilterFragments + tmp_thread_idx.x] = acc11.at(i, j);
			}

		int tmp = (output_shape.filters + OutputFilterFragments - 1) / OutputFilterFragments;
		int batch = blockIdx.x / tmp;
		int output_filter = (blockIdx.x % tmp) * OutputFilterFragments;

		__shared__ T bias_storage[OutputFilterFragments];
		if (bias != nullptr and tmp_thread_idx.y == 0)
			bias_storage[tmp_thread_idx.x] = bias[output_filter + tmp_thread_idx.x];
		__syncthreads();

		for (int row = tmp_thread_idx.y; row < OutputTileSize; row += tmp_block_dim.y)
			for (int col = 0; col < OutputTileSize; col++)
			{
				int x = OutputTileSize * blockIdx.y + row;
				int y = OutputTileSize * blockIdx.z + col;
				if (x < output_shape.height and y < output_shape.width)
				{
					T tmp = alpha1 * weight_storage[(row * OutputTileSize + col) * OutputFilterFragments + tmp_thread_idx.x] + bias_storage[tmp_thread_idx.x];

					int index = output_shape.offset_at(batch, x, y, output_filter);
					if (add != nullptr)
						tmp += alpha2 * add[index];
					tmp = Activation().forward(tmp);
					if (beta != scalar_zero<T>())
						tmp += beta * output[index];
					output[index + tmp_thread_idx.x] = tmp;
				}
			}
	}

	TensorShape get_tensor_shape(const TensorDescriptor &desc)
	{
		return TensorShape(desc);
	}
}

namespace avocado
{
	namespace backend
	{
		using namespace BACKEND_NAMESPACE;

		avStatus_t cuda_winogradFusedForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
				const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
				const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
				const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem,
				const avActivationType_t activation)
		{
			TensorShape input_shape = get_tensor_shape(getTensor(xDesc));
			TensorShape output_shape = get_tensor_shape(getTensor(yDesc));

			int batch_size = getTensor(xDesc).dimension(0);
			int tile_h = (getTensor(xDesc).dimension(1) + 7) / 8;
			int tile_w = (getTensor(xDesc).dimension(2) + 7) / 8;
			int filters_in = getTensor(xDesc).dimension(3);
			int filters_out = getTensor(yDesc).dimension(3);

			dim3 blockDim(256);
			dim3 gridDim(batch_size * ((filters_out + 63) / 64), tile_h, tile_w);
//			dim3 gridDim(1, 1, 1);
			cudaStream_t stream = getContext(context).getStream();
			getContext(context).setDevice();

			int2 padding { -1, -1 };

			switch (getTensor(yDesc).dtype())
			{
				case AVOCADO_DTYPE_FLOAT32:
				{
					float _alpha1 = getAlphaValue(alpha1);
					float _alpha2 = getAlphaValue(alpha2);
					float _beta = getBetaValue(beta);
					kernel_winograd_fused<float, ActivationLinear<float>> <<<gridDim, blockDim, 0, stream>>>(getPointer<float>(wMem), getPointer<float>(xMem),
							input_shape, getPointer<float>(yMem), output_shape, padding, _alpha1, _alpha2, _beta, getPointer<float>(bMem),
							getPointer<float>(zMem));
				}
					break;
			}
			return AVOCADO_STATUS_SUCCESS;
		}
	} /* namespace backend */
} /* namespace avocado */

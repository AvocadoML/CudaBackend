//============================================================================
// Name        : CudaBackend.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cuda_runtime_api.h>
#include <avocado/cuda_backend.h>
#include <avocado/backend/backend_descriptors.hpp>
#include <limits>

#include "../src/utilities.hpp"
#include "../src/activations.cuh"
#include "../src/winograd.hpp"
using namespace avocado::backend;

class TensorWrapper
{
private:
	avTensorDescriptor_t desc;
	avMemoryDescriptor_t mem;
public:
	TensorWrapper(std::initializer_list<int> dimensions, avDataType_t dtype, avDeviceIndex_t deviceIndex)
	{
		cudaCreateTensorDescriptor(&desc);
		cudaSetTensorDescriptor(desc, dtype, dimensions.size(), dimensions.begin());

		avSize_t size_in_bytes = getTensor(desc).sizeInBytes();
		cudaCreateMemoryDescriptor(&mem, deviceIndex, size_in_bytes);
		cudaSetMemory(cudaGetDefaultContext(deviceIndex), mem, size_in_bytes, nullptr, 0);
	}
	~TensorWrapper()
	{
		cudaDestroyTensorDescriptor(desc);
		cudaDestroyMemoryDescriptor(mem);
	}
	template<typename T>
	void fill(T value)
	{
		assert(typeOf<T>() == getTensor(desc).dtype());
		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(getTensor(desc).volume());
		for (avSize_t i = 0; i < getTensor(desc).volume(); i++)
			tmp[i] = value;
		cudaMemcpy(getPointer(mem), tmp.get(), getTensor(desc).sizeInBytes(), cudaMemcpyHostToDevice);
	}
	template<typename T>
	void set(T value, std::initializer_list<int> idx)
	{
		cudaMemcpy(getPointer<T>(mem) + getTensor(desc).getIndex(idx), &value, sizeof(T), cudaMemcpyHostToDevice);
	}
	template<typename T>
	T get(std::initializer_list<int> idx) const
	{
		T result;
		cudaMemcpy(&result, getPointer<T>(mem) + getTensor(desc).getIndex(idx), sizeof(T), cudaMemcpyDeviceToHost);
		return result;
	}
	avTensorDescriptor_t getDesc() const noexcept
	{
		return desc;
	}
	avMemoryDescriptor_t getMem() const noexcept
	{
		return mem;
	}
	template<typename T = void>
	T* data() noexcept
	{
		return getPointer<T>(mem);
	}
	template<typename T = void>
	const T* data() const noexcept
	{
		return getPointer<T>(mem);
	}
};

class ContextWrapper
{
private:
	avContextDescriptor_t desc;
public:
	ContextWrapper(avDeviceIndex_t deviceIndex)
	{
		cudaCreateContextDescriptor(&desc, deviceIndex);
	}
	~ContextWrapper()
	{
		cudaDestroyContextDescriptor(desc);
	}
	operator avContextDescriptor_t() noexcept
	{
		return desc;
	}
};

int main()
{
	std::cout << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << '\n';

//	int filters = 1024;
//	TensorWrapper weights( { filters, 3, 3, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//	TensorWrapper matrices( { 36, filters, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//
//	ContextWrapper context(0);
//
//	weights.fill(1.0f);
//	for (int i = 0; i < 3; i++)
//		for (int j = 0; j < 3; j++)
//			weights.set<float>(i * 3 + j, { 0, i, j, 0 });
//	winogradWeightTransform(context, 0, weights.getDesc(), weights.getMem(), matrices.getDesc(), matrices.getMem());
//
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//			printf("%f ", matrices.get<float>( { i * 6 + j, 0, 0 }));
//		std::cout << '\n';
//	}

	int batch_size = 128;
	int height = 20;
	int width = 20;
	int filters = 256;

	int tiles_h = (height + 3) / 4;
	int tiles_w = (width + 3) / 4;
	TensorWrapper input( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper matrices( { 36, batch_size * tiles_h * tiles_w, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	ContextWrapper context(0);

//	std::unique_ptr<float[]> storage = std::make_unique<float[]>(batch_size * height * width * filters);
//	int tmp = 0;
//	for (int i = 0; i < batch_size; i++)
//		for (int j = 0; j < height; j++)
//			for (int k = 0; k < width; k++)
//				for (int l = 0; l < filters; l++, tmp++)
//					storage[tmp] = 1.0f * i + 1.0e-1f * j + 1.0e-2f * k + 1.0e-5f * l;
//	cudaMemcpy(input.data(), storage.get(), sizeof(float) * batch_size * height * width * filters, cudaMemcpyHostToDevice);
//	winogradInputTransform(context, 0, input.getDesc(), input.getMem(), matrices.getDesc(), matrices.getMem());

	std::unique_ptr<float[]> storage = std::make_unique<float[]>(36 * batch_size * tiles_h * tiles_w * filters);
	int tmp = 0;
	for (int i = 0; i < 36; i++)
		for (int j = 0; j < batch_size * tiles_h * tiles_w; j++)
			for (int k = 0; k < filters; k++, tmp++)
				storage[tmp] = 1.0f * i + 1.0e-3f * j + 1.0e-6f * k;
	cudaMemcpy(matrices.data(), storage.get(), sizeof(float) * 36 * batch_size * tiles_h * tiles_w * filters, cudaMemcpyHostToDevice);

	winogradOutputTransform(context, -1, nullptr, matrices.getDesc(), matrices.getMem(), input.getDesc(), input.getMem(), -1, -1, nullptr, -1, -1, nullptr,
			AVOCADO_ACTIVATION_LINEAR);


//	for (int i = 0; i < 36; i++)
//		for (int j = 0; j < batch_size * 3 * 3; j++)
//			for (int k = 0; k < filters; k++)
//				if (matrices.get<float>( { i, j, k }) != i)
//				{
//					std::cout << i << " " << j << " " << k << " - " << matrices.get<float>( { i, j, k }) << '\n';
//					return 0;
//				}

//	std::cout << '\n';
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//			printf("%f ", matrices.get<float>( { i * 6 + j, 1, 0 }));
//		std::cout << '\n';
//	}

//	int batch_size = 128;
//	int filters = 256;
//	TensorWrapper matrices( { 36, 1, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//	TensorWrapper output( { 1, 4, 4, filters }, AVOCADO_DTYPE_FLOAT32, 0);
////	TensorWrapper matrices( { 36, batch_size * 5 * 5, filters }, AVOCADO_DTYPE_FLOAT32, 0);
////	TensorWrapper output( { batch_size, 20, 20, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//
//	ContextWrapper context(0);
//
//	matrices.fill(1.0f);
//	cudaConvolutionForward(context, 0, nullptr, 0, 0, 0, 0, nullptr, output.getDesc(), output.getMem(), matrices.getMem());
//
//	std::cout << '\n';
//	for (int i = 0; i < 4; i++)
//	{
//		for (int j = 0; j < 4; j++)
//			printf("%f ", output.get<float>( { 0, i, j, 0 }));
//		std::cout << '\n';
//	}

	std::cout << cudaGetErrorName(cudaGetLastError()) << '\n';
	std::cout << "END" << std::endl;
	return 0;
}

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
	template<typename T>
	T* data() noexcept
	{
		return getPointer<T>(mem);
	}
	template<typename T>
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

	int filters = 1024;
	TensorWrapper weights( { filters, 3, 3, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper matrices( { 36, filters, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	ContextWrapper context(0);

	weights.fill(1.0f);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			weights.set<float>(i * 3 + j, { 0, i, j, 0 });
	winogradWeightTransform(context, 0, weights.getDesc(), weights.getMem(), matrices.getDesc(), matrices.getMem());

	for (int i = 0; i < 6; i++)
	{
		for (int j = 0; j < 6; j++)
			printf("%f ", matrices.get<float>( { i * 6 + j, 0, 0 }));
		std::cout << '\n';
	}

//	int batch_size = 128;
//	int filters = 256;
//	TensorWrapper input( { batch_size, 20, 20, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//	TensorWrapper matrices( { 36, batch_size * 5 * 5, filters }, AVOCADO_DTYPE_FLOAT32, 0);
//
//	ContextWrapper context(0);
//
//	input.fill(1.0f);
//	cudaConvolutionForward(context, 0, nullptr, input.getDesc(), input.getMem(), 0, 0, nullptr, 0, 0, matrices.getMem());
//
//	std::cout << '\n';
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//			printf("%f ", matrices.get<float>( { i * 6 + j, 0, 110 }));
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

	std::cout << checkForErrors() << '\n';
	std::cout << "END" << std::endl;
	return 0;
}

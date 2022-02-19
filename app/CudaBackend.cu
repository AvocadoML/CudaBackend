//============================================================================
// Name        : CudaBackend.cpp
// Author      : Maciej Kozarzewski
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <cuda_runtime_api.h>
#include <CudaBackend/cuda_backend.h>
#include <backend_descriptors.hpp>
#include <limits>

#include "../src/utilities.hpp"
#include "../src/activations.cuh"
#include "../src/winograd.hpp"
using namespace avocado::backend;

#include "../src/numbers/generic_number.cuh"
#include "../src/numbers/numbers.cuh"

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

		avSize_t size_in_bytes = cuda::getTensor(desc).sizeInBytes();
		cudaCreateMemoryDescriptor(&mem, deviceIndex, size_in_bytes);
		cudaSetMemory(cudaGetDefaultContext(deviceIndex), mem, 0, size_in_bytes, nullptr, 0);
	}
	~TensorWrapper()
	{
		cudaDestroyTensorDescriptor(desc);
		cudaDestroyMemoryDescriptor(mem);
	}
	template<typename T>
	void fill(T value)
	{
		std::unique_ptr<T[]> tmp = std::make_unique<T[]>(cuda::getTensor(desc).volume());
		for (avSize_t i = 0; i < cuda::getTensor(desc).volume(); i++)
			tmp[i] = value;
		cudaMemcpy(cuda::getPointer(mem), tmp.get(), cuda::getTensor(desc).sizeInBytes(), cudaMemcpyHostToDevice);
	}
	template<typename T>
	void set(T value, std::initializer_list<int> idx)
	{
		cudaMemcpy(cuda::getPointer<T>(mem) + cuda::getTensor(desc).getIndex(idx), &value, sizeof(T), cudaMemcpyHostToDevice);
	}
	template<typename T>
	T get(std::initializer_list<int> idx) const
	{
		T result;
		cudaMemcpy(&result, cuda::getPointer<T>(mem) + cuda::getTensor(desc).getIndex(idx), sizeof(T), cudaMemcpyDeviceToHost);
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
		return cuda::getPointer<T>(mem);
	}
	template<typename T = void>
	const T* data() const noexcept
	{
		return cuda::getPointer<T>(mem);
	}
	int volume() const noexcept
	{
		return cuda::getTensor(desc).volume();
	}

	template<typename T>
	std::unique_ptr<T[]> copyToHost() const
	{
		std::unique_ptr<T[]> result = std::make_unique<T[]>(cuda::getTensor(desc).volume());
		cudaError_t status = cudaMemcpy(result.get(), cuda::getPointer(mem), cuda::getTensor(desc).sizeInBytes(), cudaMemcpyDeviceToHost);
		if (status != cudaSuccess)
			throw std::runtime_error("copyToHost");
		return result;
	}
	template<typename T>
	void copyFromHost(const std::unique_ptr<T[]> &data)
	{
		cudaError_t status = cudaMemcpy(cuda::getPointer(mem), data.get(), cuda::getTensor(desc).sizeInBytes(), cudaMemcpyHostToDevice);
		if (status != cudaSuccess)
			throw std::runtime_error("copyFromHost");
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

template<typename DataType, typename ComputeType = DataType, typename ScalingType = DataType, typename BiasType = DataType>
//void kernel_convolution_2d(const ConvolutionDescriptor &config, ScalingType alpha, const TensorDescriptor &xDesc, const DataType *xMem,
//		const TensorDescriptor &wDesc, const DataType *wMem, ScalingType beta, const TensorDescriptor &yDesc, DataType *yMem, avActivationType_t activation =
//				AVOCADO_ACTIVATION_LINEAR, ScalingType alpha2 = zero<ScalingType>(), const BiasType *bias = nullptr, const DataType *zMem = nullptr)
void kernel_convolution_2d(ScalingType alpha, const cuda::TensorDescriptor &xDesc, const DataType *xMem, const cuda::TensorDescriptor &wDesc,
		const DataType *wMem, ScalingType beta, const cuda::TensorDescriptor &yDesc, DataType *yMem, avActivationType_t activation = AVOCADO_ACTIVATION_LINEAR,
		const BiasType *bias = nullptr, ScalingType alpha2 = zero<ScalingType>(), const DataType *zMem = nullptr)
{
	const int batch_size = xDesc.dimension(0);

	const int output_filters = wDesc.dimension(0);
	const int filter_height = wDesc.dimension(1);
	const int filter_width = wDesc.dimension(2);
	const int input_filters = wDesc.dimension(3);

	const int padding_h = -1; //config.padding[0];
	const int padding_w = -1; // config.padding[1];

	const int stride_h = 1; // config.stride[0];
	const int stride_w = 1; // config.stride[1];

	const int dilation_h = 1; //config.dilation[0];
	const int dilation_w = 1; //config.dilation[1];

	const int groups = 1; //config.groups;

	const DataType padding_value = zero<DataType>(); //getScalarValue<DataType>(config.padding_value);

	if (beta == zero<ScalingType>())
	{
		for (int i = 0; i < yDesc.volume(); i++)
			yMem[i] = zero<DataType>();
	}
	for (int b = 0; b < batch_size; b++) // batch size
		for (int g = 0; g < groups; g++)
		{
			const int output_filters_group[2] = { g * output_filters / groups, (g + 1) * output_filters / groups };
			const int input_filters_group[2] = { g * input_filters / groups, (g + 1) * input_filters / groups };

			for (int out = output_filters_group[0]; out < output_filters_group[1]; out++) // output filters
				for (int out_h = 0; out_h < yDesc.dimension(1); out_h++) // output height
					for (int out_w = 0; out_w < yDesc.dimension(2); out_w++) // output width
					{
						ComputeType tmp = zero<ComputeType>();
						for (int i = 0; i < wDesc.dimension(1); i++) // kernel height
							for (int j = 0; j < wDesc.dimension(2); j++) // kernel width
							{
								int x, y;
//								if (config.mode == AVOCADO_CONVOLUTION_MODE)
								{
									x = padding_h + i * dilation_h + out_h * stride_h;
									y = padding_w + j * dilation_w + out_w * stride_w;
								}
//								else // AVOCADO_CROSS_CORRELATION_MODE
//								{
//									x = padding_h + (filter_height - 1 - i) * dilation_h + out_h * stride_h;
//									y = padding_w + (filter_width - 1 - j) * dilation_w + out_w * stride_w;
//								}
								if (x >= 0 and x < xDesc.dimension(1) and y >= 0 and y < xDesc.dimension(2))
								{
									for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
										tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, j, in })])
												* static_cast<ComputeType>(xMem[xDesc.getIndex( { b, x, y, in })]);
								}
								else
								{
									for (int in = input_filters_group[0]; in < input_filters_group[1]; in++) // input filters
										tmp += static_cast<ComputeType>(wMem[wDesc.getIndex( { out, i, j, in })]) * static_cast<ComputeType>(padding_value);
								}
							}
						ScalingType tmp2 = alpha * static_cast<ScalingType>(tmp)
								+ beta * static_cast<ScalingType>(yMem[yDesc.getIndex( { b, out_h, out_w, out })]);
						if (bias != nullptr)
							tmp2 += static_cast<ScalingType>(bias[out]);
						if (zMem != nullptr)
							tmp2 += alpha2 * static_cast<ScalingType>(zMem[yDesc.getIndex( { b, out_h, out_w, out })]);
						yMem[yDesc.getIndex( { b, out_h, out_w, out })] = tmp2;
					}
		}
}
avStatus_t refConvolutionBiasActivationForward(avContextDescriptor_t context, const avConvolutionDescriptor_t config, const void *alpha1,
		const avTensorDescriptor_t xDesc, const avMemoryDescriptor_t xMem, const avTensorDescriptor_t wDesc, const avMemoryDescriptor_t wMem,
		const avTensorDescriptor_t bDesc, const avMemoryDescriptor_t bMem, const void *alpha2, const avTensorDescriptor_t zDesc,
		const avMemoryDescriptor_t zMem, const void *beta, const avTensorDescriptor_t yDesc, avMemoryDescriptor_t yMem, const avActivationType_t activation,
		avMemoryDescriptor_t workspace)
{
//	switch (getTensor(xDesc).dtype())
//	{
//		case AVOCADO_DTYPE_FLOAT32:
//			kernel_convolution_2d(getConvolution(config), getAlphaValue(alpha1), getTensor(xDesc), getPointer<float>(xMem), getTensor(wDesc),
//					getPointer<float>(wMem), getBetaValue(beta), getTensor(yDesc), getPointer<float>(yMem), activation, getAlphaValue(alpha2),
//					getPointer<float>(bMem), getPointer<float>(zMem));
//			break;
//		default:
//			return AVOCADO_STATUS_UNSUPPORTED_DATATYPE;
//	}
	return AVOCADO_STATUS_SUCCESS;
}

void test_implicit_gemm()
{
	int batch_size = 128;
	int height = 20;
	int width = 20;
	int filters = 128;

	TensorWrapper weights( { filters, 3, 3, filters }, AVOCADO_DTYPE_INT8, 0);
	TensorWrapper bias( { filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper input( { batch_size, height, width, filters }, AVOCADO_DTYPE_INT8, 0);
	TensorWrapper output( { batch_size, height, width, filters }, AVOCADO_DTYPE_INT32, 0);

	TensorWrapper input_matrix( { batch_size * height * width, 9 * filters }, AVOCADO_DTYPE_INT8, 0);
	TensorWrapper weights_matrix( { filters, 9 * filters }, AVOCADO_DTYPE_INT8, 0);
	TensorWrapper output_matrix( { batch_size * height * width, filters }, AVOCADO_DTYPE_INT32, 0);

	ContextWrapper context(0);

	cudaGemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, input_matrix.getDesc(), input_matrix.getMem(), weights_matrix.getDesc(),
			weights_matrix.getMem(), nullptr, output_matrix.getDesc(), output_matrix.getMem());

//	std::unique_ptr<int8_t[]> storage = std::make_unique<int8_t[]>(batch_size * height * width * filters);
//	int tmp = 0;
//	for (int i = 0; i < batch_size; i++)
//		for (int j = 0; j < height; j++)
//			for (int k = 0; k < width; k++)
//				for (int l = 0; l < filters; l++, tmp++)
//					storage[tmp] = j * width + k;
//	input.copyFromHost(storage);
//
//	std::unique_ptr<int8_t[]> storage3 = std::make_unique<int8_t[]>(filters * 3 * 3 * filters);
//	tmp = 0;
//	for (int i = 0; i < filters; i++)
//		for (int j = 0; j < 3; j++)
//			for (int k = 0; k < 3; k++)
//				for (int l = 0; l < filters; l++, tmp++)
//					storage3[tmp] = 3 * j + k;
//	weights.copyFromHost(storage3);
//
//	avStatus_t status = convolution2dImplicitGemm(context, 0, nullptr, input.getDesc(), input.getMem(), weights.getDesc(), weights.getMem(), bias.getDesc(),
//			bias.getMem(), nullptr, -1, -1, nullptr, output.getDesc(), output.getMem(), AVOCADO_ACTIVATION_LINEAR);
//
//	std::unique_ptr<int[]> storage2 = output.copyToHost<int>();
//	int correct = 0, wrong = 0;
//
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//			printf("%i ", output.get<int>( { 0, i, j, 0 }));
//		printf("\n");
//	}
//	for (int i = 0; i < batch_size * height * width * filters; i++)
//	{
//		//		std::cout << i << " " << storage2[i] << "\n";
//		if (storage2[i] == 1.0f)
//			correct++;
//		else
//			wrong++;
//	}
//	std::cout << correct << " " << wrong << '\n';
}

void test_fused()
{
	int batch_size = 128;
	int height = 16;
	int width = 16;
	int filters = 128;

	TensorWrapper weights( { filters, 3, 3, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper bias( { filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper input( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper output( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	TensorWrapper input_matrix( { batch_size * height * width, 9 * filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper weights_matrix( { filters, 9 * filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper output_matrix( { batch_size * height * width, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	ContextWrapper context(0);

//	cudaGemm(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, input_matrix.getDesc(), input_matrix.getMem(), weights_matrix.getDesc(),
//			weights_matrix.getMem(), nullptr, output_matrix.getDesc(), output_matrix.getMem());

	std::unique_ptr<float[]> storage = std::make_unique<float[]>(batch_size * height * width * filters);
	int tmp = 0;
	for (int i = 0; i < batch_size; i++)
		for (int j = 0; j < height; j++)
			for (int k = 0; k < width; k++)
				for (int l = 0; l < filters; l++, tmp++)
					storage[tmp] = 1.0f * i + 1.0e-1f * j + 1.0e-2f * k + 1.0e-5f * l;
	input.copyFromHost(storage);

	std::unique_ptr<float[]> storage3 = std::make_unique<float[]>(filters * 3 * 3 * filters);
	tmp = 0;
	for (int i = 0; i < filters; i++)
		for (int j = 0; j < 3; j++)
			for (int k = 0; k < 3; k++)
				for (int l = 0; l < filters; l++, tmp++)
					storage3[tmp] = 1.0f * i + 1.0e-1f * (3 * j + k) + 1.0e-4f * l;
	weights.copyFromHost(storage3);

	avStatus_t status = cuda_winogradFusedForward(context, 0, nullptr, input.getDesc(), input.getMem(), weights.getDesc(), weights.getMem(), bias.getDesc(),
			bias.getMem(), nullptr, -1, -1, nullptr, output.getDesc(), output.getMem(), AVOCADO_ACTIVATION_LINEAR);

	std::unique_ptr<float[]> storage2 = output.copyToHost<float>();
	int correct = 0, wrong = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			printf("%f ", output.get<float>( { 0, i, j, 0 }));
		printf("\n");
	}
	for (int i = 0; i < batch_size * height * width * filters; i++)
	{
		//		std::cout << i << " " << storage2[i] << "\n";
		if (storage2[i] == 1.0f)
			correct++;
		else
			wrong++;
	}
	std::cout << correct << " " << wrong << '\n';
}

void test_nonfused()
{
	int batch_size = 32;
	int height = 15;
	int width = 15;
	int filters = 256;

	int tiles_h = (height + 3) / 4;
	int tiles_w = (width + 3) / 4;
	TensorWrapper weights( { filters, 3, 3, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper bias( { filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper input( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper output( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper output_correct( { batch_size, height, width, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	TensorWrapper weight_matrices( { 36, filters, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper input_matrices( { 36, batch_size * tiles_h * tiles_w, filters }, AVOCADO_DTYPE_FLOAT32, 0);
	TensorWrapper output_matrices( { 36, batch_size * tiles_h * tiles_w, filters }, AVOCADO_DTYPE_FLOAT32, 0);

	ContextWrapper context(0);

	auto storage = weights.copyToHost<float>();
	for (int i = 0; i < weights.volume(); i++)
		storage[i] = sin(i * 0.001f);
	weights.copyFromHost(storage);

	auto storage2 = input.copyToHost<float>();
	for (int i = 0; i < input.volume(); i++)
		storage2[i] = cos(i * 0.001f);
	input.copyFromHost(storage2);

	auto storage3 = bias.copyToHost<float>();
	for (int i = 0; i < bias.volume(); i++)
		storage3[i] = sin(i * 0.001f);
	bias.copyFromHost(storage3);

//	weights.fill(1.0f);
//	input.fill(1.0f);

//	std::unique_ptr<float[]> storage = std::make_unique<float[]>(batch_size * height * width * filters);
//	int tmp = 0;
//	for (int i = 0; i < batch_size; i++)
//		for (int j = 0; j < height; j++)
//			for (int k = 0; k < width; k++)
//				for (int l = 0; l < filters; l++, tmp++)
//					storage[tmp] = 1.0f * i + 1.0e-1f * j + 1.0e-2f * k + 1.0e-5f * l;
//	cudaMemcpy(input.data(), storage.get(), sizeof(float) * batch_size * height * width * filters, cudaMemcpyHostToDevice);

	std::unique_ptr<float[]> host_input = input.copyToHost<float>();
	std::unique_ptr<float[]> host_weights = weights.copyToHost<float>();
	std::unique_ptr<float[]> host_bias = bias.copyToHost<float>();
	std::unique_ptr<float[]> host_output = output_correct.copyToHost<float>();
	kernel_convolution_2d<float>(1.0f, cuda::getTensor(input.getDesc()), host_input.get(), cuda::getTensor(weights.getDesc()), host_weights.get(), 0.0f,
			cuda::getTensor(output_correct.getDesc()), host_output.get(), AVOCADO_ACTIVATION_LINEAR, host_bias.get());
	output_correct.copyFromHost(host_output);

	avStatus_t status = cuda_winogradWeightTransform(context, 0, weights.getDesc(), weights.getMem(), weight_matrices.getDesc(), weight_matrices.getMem());
//	for (int i = 0; i < 3; i++)
//	{
//		for (int j = 0; j < 3; j++)
//			printf("%f ", weights.get<float>( { 0, i, j, 0 }));
//		printf("\n");
//	}
//	std::cout << "----------------------------------------------------\n";
//	for (int i = 0; i < 6; i++)
//	{
//		for (int j = 0; j < 6; j++)
//			printf("%f ", weight_matrices.get<float>( { i * 6 + j, 0, 0 }));
//		printf("\n");
//	}

	status = cuda_winogradInputTransform(context, 0, input.getDesc(), input.getMem(), input_matrices.getDesc(), input_matrices.getMem());
	status = cudaGemmBatched(context, AVOCADO_GEMM_OPERATION_N, AVOCADO_GEMM_OPERATION_T, nullptr, input_matrices.getDesc(), input_matrices.getMem(),
			weight_matrices.getDesc(), weight_matrices.getMem(), nullptr, output_matrices.getDesc(), output_matrices.getMem());
	status = cuda_winogradOutputTransform(context, 0, nullptr, output_matrices.getDesc(), output_matrices.getMem(), output.getDesc(), output.getMem(),
			bias.getDesc(), bias.getMem(), nullptr, -1, -1, nullptr, AVOCADO_ACTIVATION_LINEAR);
	cudaDeviceSynchronize();
	auto storage4 = output.copyToHost<float>();
	std::cout << "CUDA code finished\n";

	double diff = 0.0;
	for (int i = 0; i < output.volume(); i++)
		diff += fabs(host_output[i] - storage4[i]);
	std::cout << "diff = " << diff / output.volume() << '\n';

//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//			printf("%f ", output_correct.get<float>( { 0, i, j, 0 }));
//		printf("\n");
//	}
//	std::cout << "----------------------------------------------------\n";
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//			printf("%f ", output.get<float>( { 0, i, j, 0 }));
//		printf("\n");
//	}
}

template<typename T>
__global__ void set_numbers(T *data, int length)
{
	for (int i = threadIdx.x; i < length; i += blockDim.x)
		numbers::Number<T>(i * 0.1f).store(data + i);
}
template<typename T>
__global__ void test_number(T *data, int length)
{
	for (int i = threadIdx.x; i < length; i += blockDim.x)
	{
		numbers::Number<T> n(data + i, length - i);
		n = sin(n);
		n.store(data + i, length - i);
	}
}

int main()
{
	std::cout << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << "." << __CUDACC_VER_BUILD__ << '\n';

	bfloat16 *data;
	cudaMalloc(&data, sizeof(bfloat16) * 1000);

	set_numbers<<<1, 256>>>(data, 1000);
	test_number<<<1, 1>>>(data, 1000);

	cudaDeviceSynchronize();

	bfloat16 host[1000];
	cudaMemcpy(host, data, sizeof(half) * 1000, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 10; i++)
		std::cout << i << " " << numbers::bfloat16_to_float(host[i]) << '\n';
	return 0;

//	test_implicit_gemm();
//	test_fused();
//	test_nonfused();

	avMemoryDescriptor_t mem;
	cudaCreateMemoryDescriptor(&mem, 1, 4000);

	avContextDescriptor_t context = cudaGetDefaultContext(1);
	avStatus_t status = cudaSetMemory(context, mem, 0, 4000, nullptr, 0);
	std::cout << status << '\n';

	cudaDestroyMemoryDescriptor(mem);

	std::cout << cudaGetErrorName(cudaGetLastError()) << '\n';
	std::cout << "END" << std::endl;
	return 0;
}

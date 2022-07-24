cmake_minimum_required(VERSION 3.18)
project(AvocadoCudaBackend VERSION 1.0.0 DESCRIPTION "CUDA backend library" LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose build type" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Choose target CUDA architecture level" FORCE)
set(PATH_TO_COMMON_UTILS "${PROJECT_SOURCE_DIR}/../CommonBackendUtils" CACHE STRING "Path to the common backend utils project")

set(commonsrc 	${PATH_TO_COMMON_UTILS}/src/descriptors/ContextDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/ConvolutionDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/DropoutDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/MemoryDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/OptimizerDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/PoolingDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/descriptors/TensorDescriptor.cpp
				${PATH_TO_COMMON_UTILS}/src/backend_utils.cpp)

set(libsrc 	src/activations.cu
		src/batchnorm.cu
		src/conv2d_implicit_gemm.cu
		src/conversions.cu
		src/convolution.cu
		src/cuda_properties.cpp
		src/descriptors.cu
		src/dropout.cu
		src/gemm.cpp
		src/im2row.cu
		src/losses.cu
		src/metrics.cu
		src/optimizers.cu
		src/pooling.cu
		src/regularizers.cu
		src/softmax.cu
		src/tensor_binary_op.cu
		src/tensor_op.cu
		src/tensor_reduction.cu
		src/tensor_unary_op.cu
		src/winograd_fused.cu
		src/winograd_nonfused.cu)
			
if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoCudaBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc} ${commonsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoCudaBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc} ${commonsrc})
endif()
			
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_COMMON_UTILS}/include")
target_include_directories(${LIBRARY_NAME} PUBLIC "/usr/local/cuda/include")
target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDA_BACKEND)
target_link_libraries(${LIBRARY_NAME} PUBLIC cudart cublas)

install(TARGETS ${LIBRARY_NAME})
install(FILES "include/Avocado/cuda_backend.h" DESTINATION include/Avocado)


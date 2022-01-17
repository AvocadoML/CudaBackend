cmake_minimum_required(VERSION 3.18)
project(AvocadoCudaBackend VERSION 1.0.0 DESCRIPTION "CPU backend library" LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" ON)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose build type" FORCE)
set(CMAKE_CUDA_ARCHITECTURES "61" CACHE STRING "Choose target CUDA architecture level" FORCE)
set(PATH_TO_MAIN_LIB "${PROJECT_SOURCE_DIR}/../Avocado" CACHE STRING "Path to the main Avocado project")

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
			src/winograd_nonfused.cu
			${PATH_TO_MAIN_LIB}/include/Avocado/backend/backend_descriptors.cpp)
			
if(BUILD_SHARED_LIBS)
  set(LIBRARY_NAME "AvocadoCudaBackend")
  add_library(${LIBRARY_NAME} SHARED ${libsrc})
  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
else()
  set(LIBRARY_NAME "AvocadoCudaBackend_static")
  add_library(${LIBRARY_NAME} STATIC ${libsrc})
endif()
			
set_target_properties(${LIBRARY_NAME} PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" "${PATH_TO_MAIN_LIB}/include/Avocado/backend")
target_include_directories(${LIBRARY_NAME} PRIVATE "${PROJECT_SOURCE_DIR}/extern")
target_compile_definitions(${LIBRARY_NAME} PRIVATE USE_CPU=0 USE_CUDA=1 USE_OPENCL=0)
target_link_libraries(${LIBRARY_NAME} PUBLIC cudart cublas)

install(TARGETS ${LIBRARY_NAME})
#install(DIRECTORY include/ DESTINATION include FILES_MATCHING PATTERN "*.h")
install(FILES "include/CudaBackend/cuda_backend.h" DESTINATION include/CudaBackend)
install(FILES "${PATH_TO_MAIN_LIB}/include/Avocado/backend/backend_defs.h" DESTINATION include/CudaBackend)


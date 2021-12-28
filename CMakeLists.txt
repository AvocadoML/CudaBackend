cmake_minimum_required(VERSION 3.18)
project(AvocadoCudaBackend VERSION 1.0.0 DESCRIPTION "CPU backend library" LANGUAGES CXX CUDA)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_WITH_OPENBLAS "Use OpenBlas for linear algebra on CPU?" ON)
option(BUILD_WITH_BLIS "Use BLIS for linear algebra on CPU?" OFF)
set(PATH_TO_MAIN_LIB "Path to the main Avocado project" "${PROJECT_SOURCE_DIR}/../Avocado")

add_library(AvocadoCudaBackend STATIC)
set_target_properties(AvocadoCudaBackend PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED YES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_include_directories(AvocadoCudaBackend PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(AvocadoCudaBackend PRIVATE "${PROJECT_SOURCE_DIR}/extern")
target_include_directories(AvocadoCudaBackend PUBLIC "${PATH_TO_MAIN_LIB}/include")

target_compile_definitions(AvocadoCudaBackend PRIVATE USE_CPU=0)
target_compile_definitions(AvocadoCudaBackend PRIVATE USE_CUDA=1)
target_compile_definitions(AvocadoCudaBackend PRIVATE USE_OPENCL=0)

target_link_libraries(AvocadoCudaBackend PUBLIC cudart)
target_link_libraries(AvocadoCudaBackend PUBLIC cublas)

add_subdirectory(src)



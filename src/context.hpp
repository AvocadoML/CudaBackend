/*
 * context.hpp
 *
 *  Created on: Nov 30, 2021
 *      Author: Maciej Kozarzewski
 */

#ifndef CONTEXT_HPP_
#define CONTEXT_HPP_

#include <avocado/cuda_backend.h>

#include <cuda_runtime_api.h>
#include <cublas_v2.h>

#include <memory>
#include <stddef.h>
#include <cassert>

#include <iostream>

namespace avocado
{
	namespace backend
	{
		class Workspace
		{
		private:
			struct ControlBlock
			{
				void* data;
				size_t *use_count;

				ControlBlock() :
						data(nullptr), use_count(new size_t)
				{
					(*use_count) = 1;
					std::cout << "default constructor " << *use_count << '\n';
				}
				ControlBlock(avSize_t size) :
						use_count(new size_t)
				{
					cudaError_t status = cudaMalloc(&data, size);
					assert(status == cudaSuccess);
					(*use_count) = 1;
					std::cout << "alloc constructor " << *use_count << '\n';
				}
				ControlBlock(const ControlBlock &other) :
						data(other.data), use_count(other.use_count)
				{
					(*use_count)++;
					std::cout << "copy constructor " << *use_count << '\n';
				}
				ControlBlock(ControlBlock &&other) noexcept
				{
					std::swap(this->data, other.data);
					std::swap(this->use_count, other.use_count);
				}
				ControlBlock& operator=(const ControlBlock &other)
				{
					this->data = other.data;
					this->use_count = other.use_count;
					(*use_count)++;
					return *this;
				}
				ControlBlock& operator=(ControlBlock &&other) noexcept
				{
					std::swap(this->data, other.data);
					std::swap(this->use_count, other.use_count);
					return *this;
				}
				~ControlBlock() noexcept
				{
					std::cout << "destructor " << *use_count << '\n';
					if (*use_count == 1)
					{
						cudaError_t status = cudaFree(data);
						assert(status == cudaSuccess);
						delete use_count;
					}
					else
						(*use_count)--;
				}
			};
			ControlBlock m_control_block;
			avSize_t m_size = 0;
		public:
			Workspace() = default;
			Workspace(avSize_t size) :
					m_control_block(size), m_size(size)
			{
			}
			avSize_t size() const noexcept
			{
				return m_size;
			}
			void* get() noexcept
			{
				return m_control_block.data;
			}
			template<typename T>
			T* get() noexcept
			{
				return reinterpret_cast<T*>(m_control_block.data);
			}
		};

		int get_device(avContext_t context) noexcept;
		cudaStream_t get_stream(avContext_t context) noexcept;
		cublasHandle_t get_handle(avContext_t context) noexcept;
		Workspace cuda_get_workspace(avContext_t context, avSize_t bytes);

	} /* namespace backend */
} /* namespace avocado */

#endif /* CONTEXT_HPP_ */

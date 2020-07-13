/* GPUDEV.hpp
 *   by Lut99
 *
 * Created:
 *   12/07/2020, 16:50:58
 * Last edited:
 *   13/07/2020, 17:42:19
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains tools useful in developing for CUDA: Specifically,
 *   it contains a CUDA-assert function and a custom CUDA-error library.
**/

#ifndef GPUDEV_HPP
#define GPUDEV_HPP

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

#ifdef CUDA

#ifdef DEBUG
#include <iostream>
#include <cstdio>

/***** DEBUG ENVIRONMENT (when enabled) *****/

/* Starts the debugging environment in a function. */
#define CUDA_DEBUG(CONTEXT) \
    const char* __DEBUG_CONTEXT__ = CONTEXT

/* Performs a general CUDA-assert. */
#define CUDA_ASSERT(MESSAGE) \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": " MESSAGE ": " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cudaMalloc. */
#define CUDA_MALLOC_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not allocate: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cuda Host-To-Device-copy. */
#define CUDA_COPYTO_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not copy to device: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cuda Device-To-Host-copy. */
#define CUDA_COPYFROM_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not copy to host: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cuda Device-To-Device-copy. */
#define CUDA_COPYON_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not copy locally on device: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cuda memset. */
#define CUDA_MEMSET_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not set memory on device: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }

/* Performs a CUDA-assert following a cuda free. */
#define CUDA_FREE_ASSERT() \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        std::cerr << "ERROR: " << __DEBUG_CONTEXT__ << ": Could not free: " << cudaGetErrorString(cudaGetLastError()) << std::endl << std::endl; \
        exit(-1); \
    }
    
#else
/***** DEBUG ENVIRONMENT (when disabled) *****/
#define CUDA_DEBUG(CONTEXT)

/* Performs a general CUDA-assert. */
#define CUDA_ASSERT(MESSAGE)

/* Performs a CUDA-assert following a cudaMalloc. */
#define CUDA_MALLOC_ASSERT()

/* Performs a CUDA-assert following a cuda Host-To-Device-copy. */
#define CUDA_COPYTO_ASSERT()

/* Performs a CUDA-assert following a cuda Device-To-Host-copy. */
#define CUDA_COPYFROM_ASSERT()

/* Performs a CUDA-assert following a cuda free. */
#define CUDA_FREE_ASSERT()

#endif

#endif

#endif

/* OSTREAM.cpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 16:22:52
 * Last edited:
 *   16/07/2020, 14:28:37
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The ostream class serves as a RayTracer-only baseclass for output
 *   stream operations. While the class itself is CPU-only, it features
 *   abilities to spawn an ostream_gpu child on the GPU which collects
 *   prints and then streams them back to the CPU once ostream::sync() is
 *   called. Note that while CPU-side printing is instant, GPU-print only
 *   happens once sync() is called and therefore at the end of a kernel.
**/

#include "ostream.hpp"

using namespace std;
using namespace RayTracer;


/* Constructor for the ostream class. Wraps any std::ostream object to write to. The maximum buffer size of the GPU can be given as option, but must be a multiple of 4. */
RayTracer::ostream::ostream(std::ostream& os, size_t max_size) :
    os(os)
{
    #ifdef CUDA
    CUDA_DEBUG("ostream constructor");

    // Fill in the sizes
    this->n_chars = 0;
    this->max_size = max_size;
    
    // Allocate the required space on the GPU
    cudaMalloc((void**) &(this->buffer), sizeof(unsigned char) * this->max_size);
    CUDA_MALLOC_ASSERT();

    // Set everything to zero
    cudaMemset((void*) this->buffer, 0, sizeof(unsigned char) * this->max_size);
    CUDA_MEMSET_ASSERT();
    #else
    // Discard the max_size as not used
    (void) max_size;
    #endif
}

RayTracer::ostream::ostream(const RayTracer::ostream& other) :
    os(other.os)
{
    #ifdef CUDA
    CUDA_DEBUG("ostream copy constructor");

    // Also copy the sizes
    this->n_chars = other.n_chars;
    this->max_size = other.max_size;

    // Copy the data
    cudaMalloc((void**) &(this->buffer), sizeof(unsigned char) * this->max_size);
    CUDA_MALLOC_ASSERT();
    cudaMemcpy((void*) this->buffer, (void*) other.buffer, sizeof(unsigned char) * this->max_size, cudaMemcpyDeviceToDevice);
    CUDA_COPYON_ASSERT();
    #endif
}

RayTracer::ostream::ostream(RayTracer::ostream&& other) :
    os(other.os)
{
    #ifdef CUDA
    // Also copy the sizes
    this->n_chars = other.n_chars;
    this->max_size = other.max_size;

    // Steal the GPU-side data
    this->buffer = other.buffer;
    other.buffer = nullptr;
    #endif
}

RayTracer::ostream::~ostream() {
    #ifdef CUDA
    CUDA_DEBUG("ostream destructor");

    // Deallocate the memory if not stolen or anything
    if (this->buffer != nullptr) {
        cudaFree((void*) this->buffer);
        CUDA_FREE_ASSERT();
    }
    #endif
}



#ifdef CUDA
__device__ void write(size_t n, const char* str) {
    int to_write = 0;
    for (size_t i = 0; i < n; i++) {
        // Fetch the character
        char c = str[i];

        // Put at the correct place in the to_write int
        int index = i % 4;
        to_write |= ((unsigned int) value) << (8 * index);
        if (i == n - 1 || index == 3) {
            // Send the index off the the buffer
            ((unsigned int*) this->buffer)[i / 4] = to_write;
            to_write = 0;
        }
    }
    this->n_chars += n;
}

__device__ void write(const char* str) {
    int to_write = 0;
    for (size_t i = 0; ; i++) {
        // Fetch the character
        char c = str[i];

        // Put at the correct place in the to_write int
        int index = i % 4;
        to_write |= ((unsigned int) value) << (8 * index);
        if (c == '\0' || index == 3) {
            // Send the index off the the buffer
            ((unsigned int*) this->buffer)[i / 4] = to_write;
            to_write = 0;
        }

        // Stop clause
        if (c == '\0') { break; }
    }
}
#endif



void RayTracer::ostream::sync() {
    #ifdef CUDA
    CUDA_DEBUG("ostream sync");

    // Get everything back from the GPU.
    unsigned char cpu_buffer[this->n_chars];
    cudaMemcpy((void*) cpu_buffer, (void*) this->buffer, sizeof(unsigned char) * this->n_chars, cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Write the characters one-by-one, handling special characters as well.
    for (size_t i = 0; i < this->n_chars; i++) {
        char c = (char) cpu_buffer[i];
        
        switch(c) {
            default:
                this->os << c;
                break;
        }
    }

    // Reset the GPU-side buffer
    cudaMemset((void*) this->buffer, 0, sizeof(unsigned char) * this->n_chars);
    CUDA_MEMSET_ASSERT();
    this->n_chars = 0;
    #endif
}



RayTracer::ostream& RayTracer::ostream::operator=(RayTracer::ostream&& other) {

}

void RayTracer::swap(RayTracer::ostream& os1, RayTracer::ostream& os2) {

}

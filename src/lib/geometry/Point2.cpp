/* POINT2.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:03:11
 * Last edited:
 *   13/07/2020, 14:30:56
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Point2 class represents a 2-dimensional point not in the world but in
 *   the output frame. To this end, the Point2 class stores positive integers
 *   only rather than decimals who can also be negative.
**/

#include <cstdio>

#include "GPUTools.hpp"

#include "Point2.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE Point2::Point2() :
    x(0),
    y(0)
{}

HOST_DEVICE Point2::Point2(size_t x, size_t y) :
    x(x),
    y(y)
{}

HOST_DEVICE Point2::Point2(const Point2& other) :
    x(other.x),
    y(other.y)
{}

HOST_DEVICE Point2::Point2(Point2&& other) :
    x(other.x),
    y(other.y)
{}



#ifdef CUDA
Point2* Point2::GPU_create(void* ptr) {
    CUDA_DEBUG("Point2 GPU-default constructor");

    // Create a template Point2 to copy
    Point2 temp;

    // Allocate space if needed
    Point2* ptr_gpu = (Point2*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Point2));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the class
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Point2), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Done, return
    return ptr_gpu;
}

Point2* Point2::GPU_create(size_t x, size_t y, void* ptr) {
    CUDA_DEBUG("Point2 GPU-constructor");

    // Create a template Point2 to copy
    Point2 temp(x, y);

    // Allocate space if needed
    Point2* ptr_gpu = (Point2*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Point2));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the class
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Point2), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Done, return
    return ptr_gpu;
}

Point2* Point2::GPU_create(const Point2& other, void* ptr) {
    CUDA_DEBUG("Point2 GPU-copy constructor");

    // Create a template Point2 to copy
    Point2 temp(other);

    // Allocate space if needed
    Point2* ptr_gpu = (Point2*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Point2));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the class
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Point2), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Done, return
    return ptr_gpu;
}

Point2 Point2::GPU_copy(Point2* ptr_gpu) {
    CUDA_DEBUG("Point2 GPU-copy");

    // Create a CPU-side target
    Point2 result;

    // Copy the GPU-side object into it
    cudaMemcpy((void*) &result, (void*) ptr_gpu, sizeof(Point2), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Done, return
    return result;
}

void Point2::GPU_free(Point2* ptr_gpu) {
    CUDA_DEBUG("Point2 GPU-destructor");

    // Simply free the pointer
    cudaFree(ptr_gpu);
    CUDA_FREE_ASSERT();
}
#endif



HOST_DEVICE Point2& Point2::operator+=(const Point2& other) {
    this->x += other.x;
    this->y += other.y;
    return *this;
}



HOST_DEVICE Point2& Point2::operator-=(const Point2& other) {
    this->x -= other.x;
    this->y -= other.y;
    return *this;
}



HOST_DEVICE Point2& Point2::operator*=(const Point2& other) {
    this->x *= other.x;
    this->y *= other.y;
    return *this;
}



HOST_DEVICE Point2& Point2::operator/=(const Point2& other) {
    this->x /= other.x;
    this->y /= other.y;
    return *this;
}



HOST_DEVICE size_t Point2::operator[](size_t index) const {
    if (index == 0) { return this->x; }
    else if (index == 1) { return this->y; }
    
    // Else, return 0 but print a warning
    printf("ERROR: size_t Point2::operator[](size_t index) const: Index %lu is out of bounds for Point2 with size 2.\n", index);
    return 0;
}

HOST_DEVICE size_t& Point2::operator[](size_t index) {
    if (index == 0) { return this->x; }
    else if (index == 1) { return this->y; }
    
    // Else, return x but print a warning
    printf("ERROR: size_t& Point2::operator[](size_t index): Index %lu is out of bounds for Point2 with size 2.\n", index);
    return this->x;
}



HOST_DEVICE Point2& Point2::operator=(Point2&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(Point2& p1, Point2& p2) {
    // Swap the coordinates
    swap(p1.x, p2.x);
    swap(p1.y, p2.y);
}

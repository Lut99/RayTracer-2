/* VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:09:06 PM
 * Last edited:
 *   12/07/2020, 17:34:15
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Vec3-class is where to look for when needing linear algebra
 *   regarding three-dimensional vectors. This particular file focusses on
 *   the CPU-side, but there is an equivalent GPU-side library available as
 *   well
**/

#include "Vec3.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE Vec3::Vec3() :
    x(0),
    y(0),
    z(0)
{}

HOST_DEVICE Vec3::Vec3(double x, double y, double z) :
    x(x),
    y(y),
    z(z)
{}

HOST_DEVICE Vec3::Vec3(const Vec3& other) :
    x(other.x),
    y(other.y),
    z(other.z)
{}

HOST_DEVICE Vec3::Vec3(Vec3&& other) :
    x(other.x),
    y(other.y),
    z(other.z)
{}



#ifdef CUDA
Vec3* Vec3::GPU_create(void* ptr) {
    CUDA_DEBUG("Vec3 GPU-default constructor");

    // Create a CPU-side template
    Vec3 temp;

    // Allocate new memory if that is needed
    Vec3* ptr_gpu = (Vec3*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Vec3));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the default vector
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Vec3), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Vec3* Vec3::GPU_create(double x, double y, double z, void* ptr) {
    CUDA_DEBUG("Vec3 GPU-constructor");

    // Create a CPU-side template
    Vec3 temp(x, y, z);

    // Allocate new memory if that is needed
    Vec3* ptr_gpu = (Vec3*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Vec3));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the default vector
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Vec3), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Vec3* Vec3::GPU_create(const Vec3& other, void* ptr) {
    CUDA_DEBUG("Vec3 GPU-copy constructor");

    // Allocate new memory if that is needed
    Vec3* ptr_gpu = (Vec3*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Vec3));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the given vector
    cudaMemcpy((void*) ptr_gpu, (void*) &other, sizeof(Vec3), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Vec3 Vec3::GPU_copy(Vec3* ptr_gpu) {
    CUDA_DEBUG("Vec3 GPU-copy");

    // Create a Vec3 buffer on the CPU
    Vec3 result;

    // Copy the GPU-side one into it
    cudaMemcpy((void*) &result, (void*) ptr_gpu, sizeof(Vec3), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Return the new Vec as copy of the buffer
    return result;
}

void Vec3::GPU_free(Vec3* ptr_gpu) {
    CUDA_DEBUG("Vec3 GPU-destructor");

    // Deallocate the given gpu-side pointer
    cudaFree((void*) ptr_gpu);
    CUDA_FREE_ASSERT();
}
#endif



HOST_DEVICE Vec3& Vec3::operator+=(double c) {
    this->x += c;
    this->y += c;
    this->z += c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator+=(const Vec3& other) {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator-=(double c) {
    this->x -= c;
    this->y -= c;
    this->z -= c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator-=(const Vec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator*=(double c) {
    this->x *= c;
    this->y *= c;
    this->z *= c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator*=(const Vec3& other) {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator=(Vec3&& other) {
    // Only swap if not ourselves
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(Vec3& v1, Vec3& v2) {
    // Simply swap x, y and z; the referenced items will follow, and we needn't change whether it's external or not.
    double t = v1.x;
    v1.x = v2.x;
    v2.x = t;

    t = v1.y;
    v1.y = v2.y;
    v2.y = t;

    t = v1.z;
    v1.z = v2.z;
    v2.z = t;
}



HOST_DEVICE double Vec3::operator[](const size_t i) const {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double Vec3::operator[](const size_t i) const: Index %lu is out of bounds for vector of length 3.", i);
    return -INFINITY;
}

HOST_DEVICE double& Vec3::operator[](const size_t i) {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double& Vec3::operator[](const size_t i): Index %lu is out of bounds for vector of length 3.", i);
    return this->x;
}



/* Allows the vector to be printed to a stream. */
std::ostream& RayTracer::operator<<(std::ostream& os, const Vec3& vec) {
    return os << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
}

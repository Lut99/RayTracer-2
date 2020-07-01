/* GVEC 3.cu
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 2:19:02 PM
 * Last edited:
 *   7/1/2020, 5:40:27 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file provides the same math as Vec3 does, except that all its
 *   functions live on the GPU. Additionally, it also provides the means to
 *   copy a given Vec3 to the GPU as GVec3 and to copy GVec3 back as Vec3.
**/

#include <cstdio>

#include "GVec3.hpp"
#include "GPUTools.hpp"

using namespace std;
using namespace RayTracer;


__device__ GVec3::GVec3() :
    x(this->dx),
    y(this->dy),
    z(this->dz)
{
    this->dx = 0;
    this->dy = 0;
    this->dz = 0;
    this->is_local = true;
}

__device__ GVec3::GVec3(double x, double y, double z) :
    x(this->dx),
    y(this->dy),
    z(this->dz)
{
    this->dx = x;
    this->dy = y;
    this->dz = z;
    this->is_local = true;
}

__device__ GVec3::GVec3(void* ptr) :
    x(((double*) ptr)[0]),
    y(((double*) ptr)[1]),
    z(((double*) ptr)[2])
{
    // Link the storage class & doubles
    this->data = (double*) ptr;
    this->is_local = false;
}

__device__ GVec3::GVec3(const GVec3& other) :
    x(this->dx),
    y(this->dy),
    z(this->dz)
{
    // Always copy to a local data
    this->dx = other.x;
    this->dy = other.y;
    this->dz = other.z;
    this->is_local = true;
}

__device__ GVec3::GVec3(GVec3&& other) :
    x(this->dx),
    y(this->dy),
    z(this->dz)
{    
    // Always copy to a local data
    this->dx = other.x;
    this->dy = other.y;
    this->dz = other.z;
    this->is_local = true;
}



__device__ GVec3& GVec3::operator+=(const double c) {
    this->x += c;
    this->y += c;
    this->z += c;
    return *this;
}

__device__ GVec3& GVec3::operator+=(const GVec3& other) {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
}



__device__ GVec3& GVec3::operator-=(const double c) {
    this->x -= c;
    this->y -= c;
    this->z -= c;
    return *this;
}

__device__ GVec3& GVec3::operator-=(const GVec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



__device__ GVec3& GVec3::operator*=(const double c) {
    this->x *= c;
    this->y *= c;
    this->z *= c;
    return *this;
}

__device__ GVec3& GVec3::operator*=(const GVec3& other) {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
}



__device__ double GVec3::operator[](const size_t i) const {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double GVec3::operator[](const size_t i) const: Index %lu is out of bounds for vector of length 3.", i);
    return -INFINITY;
}

__device__ double& GVec3::operator[](const size_t i) {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double& GVec3::operator[](const size_t i): Index %lu is out of bounds for vector of length 3.", i);
    return this->x;
}



__device__ GVec3& GVec3::operator=(GVec3 other) {
    // Only do if not ourselves
    if (this != &other) {
        // Swap 'em
        swap(*this, other);
    }
    return *this;
}

__device__ GVec3& GVec3::operator=(GVec3&& other) {
    // Only do if not ourselves
    if (this != &other) {
        // Swap 'em
        swap(*this, other);
    }
    return *this;
}

__device__ void RayTracer::swap(GVec3& vec1, GVec3& vec2) {
    // Only swap the values for x, y and z; the referenced objects will follow
    double t = vec1.x;
    vec1.x = vec2.x;
    vec2.x = t;

    t = vec1.y;
    vec1.y = vec2.y;
    vec2.y = t;

    t = vec1.z;
    vec1.z = vec2.z;
    vec2.z = t;
}



__host__ void* GVec3::toGPU(const Vec3& vec) {
    // Allocate a brief bit of memory
    double temp[3] = { vec.x, vec.y, vec.z };
    
    // Allocate GPU-memory
    void* ptr;
    cudaMalloc(&ptr, sizeof(double) * 3);

    // Copy to the GPU
    cudaMemcpy(ptr, (void*) temp, sizeof(double) * 3, cudaMemcpyHostToDevice);

    return ptr;
}

__host__ Vec3 GVec3::fromGPU(void* ptr) {
    // Allocate a brief bit of memory
    double temp[3];

    // Copy back from the GPU
    cudaMemcpy((void*) temp, ptr, sizeof(double) * 3, cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(ptr);

    // Create a new Vec3 object
    return Vec3(temp[0], temp[1], temp[2]);
}



__device__ void GVec3::print() const {
    printf("[%f, %f, %f]", this->x, this->y, this->z);
}



__device__ GVec3 RayTracer::exp(const GVec3& vec) {
    return GVec3(exp(vec.x), exp(vec.y), exp(vec.z));
}

__device__ GVec3 RayTracer::sqrt(const GVec3& vec) {
    return GVec3(sqrt(vec.x), sqrt(vec.y), sqrt(vec.z));
}

__device__ GVec3 RayTracer::pow(const GVec3& vec, const double c) {
    return GVec3(pow(vec.x, c), pow(vec.y, c), pow(vec.z, c));
}

__device__ GVec3 RayTracer::fabs(const GVec3& vec) {
    return GVec3(fabs(vec.x), fabs(vec.y), fabs(vec.z));
}

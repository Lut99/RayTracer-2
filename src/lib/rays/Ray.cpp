/* RAY.cpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:09:47
 * Last edited:
 *   07/07/2020, 17:32:28
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Ray class provides access to a single ray, which has an origin and
 *   a direction. Additionally, it supports GPU-related activities and
 *   therefore also the ability to use external managed memory.
**/

#include "Ray.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE Ray::Ray() :
    origin(0, 0, 0),
    direction(0, 0, 0)
{}

HOST_DEVICE Ray::Ray(const Point3& origin, const Vec3& direction) :
    origin(origin),
    direction(direction)
{}





#ifdef CUDA
Ray* Ray::GPU_create(void* ptr) {
    // Create a template Ray to copy
    Ray temp;

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);

    // Return the pointer
    return ptr_gpu;
}

Ray* Ray::GPU_create(const Point3& origin, const Vec3& direction, void* ptr) {
    // Create a template Ray to copy
    Ray temp(origin, direction);

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);

    // Return the pointer
    return ptr_gpu;
}

Ray* Ray::GPU_create(const Ray& other, void* ptr) {
    // Create a template Ray to copy
    Ray temp(other);

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);

    // Return the pointer
    return ptr_gpu;
}

Ray Ray::GPU_copy(Ray* ptr_gpu) {
    // Allocate a local Ray to copy over
    Ray result;

    // Copy the stuff from the given pointer back
    cudaMemcpy((void*) &result, (void*) ptr_gpu, sizeof(Ray), cudaMemcpyHostToDevice);

    // Return the Ray
    return result;
}

void Ray::GPU_free(Ray* ptr_gpu) {
    // Free the given pointer
    cudaFree(ptr_gpu);
}
#endif



HOST_DEVICE Ray& Ray::operator=(Ray other) {
    swap(*this, other);
    return *this;
}

HOST_DEVICE Ray& Ray::operator=(Ray&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(Ray& r1, Ray& r2) {
    // Simply swap the two vectors
    swap(r1.origin, r2.origin);
    swap(r1.direction, r2.direction);
}

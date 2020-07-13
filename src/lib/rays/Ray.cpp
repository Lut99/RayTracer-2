/* RAY.cpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:09:47
 * Last edited:
 *   13/07/2020, 15:28:24
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Ray class provides access to a single ray, which has an origin and
 *   a direction. Additionally, it supports GPU-related activities and
 *   therefore also the ability to use external managed memory.
**/

#include <climits>

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

HOST_DEVICE Ray::Ray(const Ray& other) :
    origin(other.origin),
    direction(other.direction)
{}

HOST_DEVICE Ray::Ray(Ray&& other) :
    origin(other.origin),
    direction(other.direction)
{}





#ifdef CUDA
Ray* Ray::GPU_create(void* ptr) {
    CUDA_DEBUG("Ray GPU-default constructor");

    // Create a template Ray to copy
    Ray temp;

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Ray* Ray::GPU_create(const Point3& origin, const Vec3& direction, void* ptr) {
    CUDA_DEBUG("Ray GPU-constructor 1");

    // Create a template Ray to copy
    Ray temp(origin, direction);

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Ray* Ray::GPU_create(const Ray& other, void* ptr) {
    CUDA_DEBUG("Ray GPU-copy constructor");

    // Create a template Ray to copy
    Ray temp(other);

    // If needed, allocate the required space
    Ray* ptr_gpu = (Ray*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Ray));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the Ray object over, which automatically includes any and all vectors
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Ray), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Ray Ray::GPU_copy(Ray* ptr_gpu) {
    CUDA_DEBUG("Ray GPU-copy");

    // Allocate a local Ray to copy over
    Ray result;

    // Copy the stuff from the given pointer back
    cudaMemcpy((void*) &result, (void*) ptr_gpu, sizeof(Ray), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Return the Ray
    return result;
}

void Ray::GPU_free(Ray* ptr_gpu) {
    CUDA_DEBUG("Ray GPU-destructor");

    // Free the given pointer
    cudaFree(ptr_gpu);
    CUDA_FREE_ASSERT();
}
#endif



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



ostream& RayTracer::operator<<(ostream& os, const Ray& ray) {
    os << "[Ray : orig " << ray.origin << ", dir " << ray.direction << "]";
    return os;
}

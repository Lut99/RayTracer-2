/* RAY.cpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:09:47
 * Last edited:
 *   05/07/2020, 17:49:21
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
__device__ Ray::Ray(void* ptr) :
    origin(ptr),
    direction((void*) ((double*) ptr + 3))
{}
#endif



#ifdef CUDA
void* Ray::toGPU(void* data) const {
    // Allocate enough space if needed
    if (data == nullptr) {
        cudaMalloc(&data, sizeof(double) * 6);
    }

    // Copy the origin
    this->origin.toGPU(data);
    // Copy the direction
    this->direction.toGPU((void*) ((double*) data + 3));

    // Return the pointer
    return data;
}

Ray Ray::fromGPU(void* ptr) {
    // Create a new origin from the pointer
    Point3 origin = Point3::fromGPU(ptr);
    // Create a new direction from the pointer
    Vec3 direction = Vec3::fromGPU((void*) ((double*) data + 3));

    // Return as new Ray
    return Ray(origin, direction);
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

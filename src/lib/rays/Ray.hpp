/* RAY.hpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:10:06
 * Last edited:
 *   07/07/2020, 17:36:21
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Ray class provides access to a single ray, which has an origin and
 *   a direction. Additionally, it supports GPU-related activities and
 *   therefore also the ability to use external managed memory.
**/

#ifndef RAY_HPP
#define RAY_HPP

#include "Vec3.hpp"


#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    class Ray {
    public:
        /* The direction of the Origin. */
        Point3 origin;
        /* The direction of the Ray. */
        Vec3 direction;

        /* Default constructor for the Ray class, which initializes it originating from (0, 0, 0) and going to (0, 0, 0). */
        HOST_DEVICE Ray();
        /* Constructor for the Ray class which takes an origin and a direction, which are stored locally. */
        HOST_DEVICE Ray(const Point3& origin, const Vec3& direction);

        #ifdef CUDA
        /* Initializes a default GPU-side Ray-object from the CPU. If ptr != nullptr, does not allocate but only copies to the given ptr. */
        static Ray* GPU_create(void* ptr = nullptr);
        /* Initializes a GPU-side Ray-object from the CPU with given origin and direction. If ptr != nullptr, does not allocate but only copies to the given ptr. */
        static Ray* GPU_create(const Point3& origin, const Vec3& direction, void* ptr = nullptr);
        /* Initializes a GPU-side Ray-object from the CPU which is a copy of given Ray-object. If ptr != nullptr, does not allocate but only copies to the given ptr. */
        static Ray* GPU_create(const Ray& other, void* ptr = nullptr);
        /* Copies relevant data from a Ray on the GPU back to the CPU as a new stack-allocated object. */
        static Ray GPU_copy(Ray* ptr_gpu);
        /* CPU-side destructor for the GPU-side Ray. */
        static void GPU_free(Ray* ptr_gpu);
        #endif

        /* Returns a point on the Ray with t distance from the origin. */
        HOST_DEVICE inline Point3 at(double t) const { return this->origin + t * this->direction; }

        /* Copy assignment operator for the Ray class. */
        HOST_DEVICE Ray& operator=(Ray other);
        /* Move assignment operator for the Ray class. */
        HOST_DEVICE Ray& operator=(Ray&& other);
        /* Swap operator for the Ray class. */
        friend HOST_DEVICE void swap(Ray& r1, Ray& r2);
    };

    /* Swap operator for the Ray class. */
    HOST_DEVICE void swap(Ray& r1, Ray& r2);
}

#endif

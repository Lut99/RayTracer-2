/* RAY.hpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:10:06
 * Last edited:
 *   05/07/2020, 17:41:01
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
        /* Constructor for the Ray class which initializes the internal vectors using the external memory. */
        __device__ Ray(void* ptr);
        #endif

        #ifdef CUDA
        /* Copies the Ray and associated vectors to the GPU. Optionally takes a point to GPU-allocated data to copy everything there. */
        void* toGPU(void* data = nullptr) const;
        /* Copies the Ray and associated vectors back from the GPU. */
        static Ray fromGPU(void* ptr);
        /* Returns the amount of bytes that need to be copied to the GPU. */
        inline size_t copy_size() const { return this->origin.copy_size() + this->direction.copy_size(); }
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

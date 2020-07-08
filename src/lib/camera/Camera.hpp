/* CAMERA.hpp
 *   by Lut99
 *
 * Created:
 *   08/07/2020, 22:21:48
 * Last edited:
 *   08/07/2020, 22:21:48
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Camera class is used on the CPU to cast Rays, which can then be
 *   bounced and eventually traced by an appropriate renderer. For maximum
 *   speedup, it supports two modes of casting: single-ray casting and
 *   batch-ray casting (the latter is particularly useful for the GPU).
 *   Additionally, like a real-life camera, it can be moved around and
 *   rotated.
**/

#ifndef CAMERA_HPP
#define CAMERA_HPP

#include "Vec3.hpp"
#include "Ray.hpp"

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    class Camera {
    private:
        /* The origin of the Camera. */
        Point3 origin;
        /* The lower-left corner of the rectangle which will be used to direct the Rays. */
        Point3 lower_left;
        /* The width of the rectangle that will be used to direct the Rays. */
        Vec3 horizontal;
        /* The height of the rectangle that will be used to direct the Rays. */
        Vec3 vertical;

        /* The width of the target frame (in pixels). */
        size_t frame_width;
        /* The height of the target frame (in pixels). */
        size_t frame_height;

    public:
        /* The width (in the world) of the Camera rectangle. */
        double viewport_width;
        /* The height (in the world) of the Camera rectangle. */
        double viewport_height;
        /* The distance between the origin and the rectangle of the Camera. */
        double focal_length;
        /* The point where the Camera stands. */
        Point3 lookfrom;
        /* The point where the Camera looks to. */
        Point3 lookat;
        /* The vector pointing up in the frame relative to the axis of the 3D-world. */
        Vec3 up;
        /* The vertical field-of-view, in degrees. */
        double vfov;

        /* Constructor for the Camera class. Takes a lot of information, which is stored in publicly available members. The Camera automatically calls recompute() to also initialize the inner members. */
        HOST_DEVICE Camera(
            double viewport_width,
            double viewport_height,
            double focal_length,
            Point3 lookfrom,
            Point3 lookat,
            Vec3 up,
            double vfov,
            size_t frame_width,
            size_t frame_height
        );

        #ifdef CUDA
        /* CPU-side constructor for a GPU-side Camera class. Takes a lot of information, which is stored in publicly available members. The Camera automatically calls recompute() to also initialize the inner members. If ptr == nullptr, does not allocate but instead only copies to the memory location pionted to. */
        static Camera* GPU_create(
            double viewport_width,
            double viewport_height,
            double focal_length,
            Point3 lookfrom,
            Point3 lookat,
            Vec3 up,
            double vfov,
            size_t frame_width,
            size_t frame_height
            void* ptr = nullptr
        );
        /* CPU-side constructor for a GPU-side Camera class. Takes another Camera object and copies it. If ptr == nullptr, does not allocate but instead only copies to the memory location pionted to. */
        static Camera* GPU_create(const Camera& other, void* ptr = nullptr);
        /* Copies a GPU-side Camera object to the CPU as a stack-allocated, new object. */
        static Camera GPU_copy(Camera* ptr_gpu);
        /* CPU-side deconstructor for a GPU-side Camera object. */
        static void GPU_free(Camera* ptr_gpu);
        #endif

        /* Re-computes the Camera class based on the origin, target, up and vfov that is stored in the class. */
        HOST_DEVICE void recompute();

        /* Casts a single ray on the given index (with a random offset). */
        HOST_DEVICE Ray cast(size_t x, size_t y) const;
    };
}

#endif

/* CAMERA.cpp
 *   by Lut99
 *
 * Created:
 *   08/07/2020, 22:21:30
 * Last edited:
 *   08/07/2020, 22:21:30
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

#include "Camera.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE Camera::Camera(Point3 origin, Point3 target, Vec3 up, double vfov, size_t width, size_t height, size_t n_rays) :
    width(width),
    height(height),
    n_rays(n_rays),
    total_rays(width * height * n_rays)
{

}



#ifdef CUDA
Camera* Camera::GPU_create(Point3 origin, Point3 target, Vec3 up, double vfov, size_t width, size_t height, size_t n_rays, void* ptr) {

}

Camera* Camera::GPU_create(const Camera& other, void* ptr) {

}

Camera Camera::GPU_copy(Camera* ptr_gpu) {

}

void Camera::GPU_free(Camera* ptr_gpu) {

}
#endif



HOST_DEVICE Ray Camera::cast(size_t x, size_t y) const {

}

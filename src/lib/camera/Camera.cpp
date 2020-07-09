/* CAMERA.cpp
 *   by Lut99
 *
 * Created:
 *   08/07/2020, 22:21:30
 * Last edited:
 *   09/07/2020, 17:58:48
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

#include "GPUTools.hpp"

#include "Camera.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE Camera::Camera(Point3 lookfrom, Point3 lookat, Vec3 up, double vfov, size_t frame_width, size_t frame_height, double viewport_width, double viewport_height, double focal_length) :
    frame_width(frame_width),
    frame_height(frame_height),
    viewport_width(viewport_width),
    viewport_height(viewport_height),
    focal_length(focal_length),
    lookfrom(lookfrom),
    lookat(lookat),
    up(up),
    vfov(vfov)
{
    this->recompute();
}



#ifdef CUDA
Camera* Camera::GPU_create(Point3 lookfrom, Point3 lookat, Vec3 up, double vfov, size_t frame_width, size_t frame_height, double viewport_width, double viewport_height, double focal_length, void* ptr) {
    // Create a CPU-side Camera template to copy
    Camera temp(lookfrom, lookat, up, vfov, frame_width, frame_height, viewport_width, viewport_height, focal_length);

    // Allocate CPU memory if needed
    Camera* ptr_gpu = (Camera*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Camera));
    }

    // Copy the data
    cudaMemcpy((void*) ptr_gpu, &temp, sizeof(Camera), cudaMemcpyHostToDevice);

    // Return the pointer
    return ptr_gpu;
}

Camera* Camera::GPU_create(const Camera& other, void* ptr) {
    // Create a CPU-side Camera template to copy
    Camera temp(other);

    // Allocate CPU memory if needed
    Camera* ptr_gpu = (Camera*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Camera));
    }

    // Copy the data
    cudaMemcpy((void*) ptr_gpu, &temp, sizeof(Camera), cudaMemcpyHostToDevice);

    // Return the pointer
    return ptr_gpu;
}

Camera Camera::GPU_copy(Camera* ptr_gpu) {
    // Create a CPU-side target memory on the stack
    Camera result;

    // Copy from the GPU into it
    cudaMemcpy((void*) &result, (void*) ptr_gpu, sizeof(Camera), cudaMemcpyDeviceToHost);

    // Return the object
    return result;
}

void Camera::GPU_free(Camera* ptr_gpu) {
    // Call cuda's free
    cudaFree((void*) ptr_gpu);
}
#endif



HOST_DEVICE void Camera::recompute() {
    // Compute the origin & rectangle of the camera
    Vec3 w = (this->lookfrom - this->lookat).normalize();
    Vec3 u = (cross(this->up, w)).normalize();
    Vec3 v = cross(w, u);

    this->origin = this->lookfrom;
    this->horizontal = this->viewport_width * u;
    this->vertical = this->viewport_height * v;
    this->lower_left = this->origin - this->horizontal / 2 - this->vertical / 2 - w;
}



HOST_DEVICE Ray Camera::cast(size_t x, size_t y) const {
    // Cast a ray through the given pixel at x and y
    double s = x / (double) this->frame_width;
    double t = y / (double) this->frame_height;
    return Ray(this->origin, this->lower_left + s * this->horizontal + t * this->vertical - this->origin);
}



HOST_DEVICE void RayTracer::swap(Camera& c1, Camera& c2) {
    // Swap everything
    swap(c1.origin, c2.origin);
    swap(c1.lower_left, c2.lower_left);
    swap(c1.horizontal, c2.horizontal);
    swap(c1.vertical, c2.vertical);
    swap(c1.frame_width, c2.frame_width);
    swap(c1.frame_height, c2.frame_height);
    swap(c1.viewport_width, c2.viewport_width);
    swap(c1.viewport_height, c2.viewport_height);
    swap(c1.focal_length, c2.focal_length);
    swap(c1.lookfrom, c2.lookfrom);
    swap(c1.lookat, c2.lookat);
    swap(c1.up, c2.up);
    swap(c1.vfov, c2.vfov);
}

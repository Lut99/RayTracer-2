/* RAY BATCH ITERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:21:15
 * Last edited:
 *   13/07/2020, 15:19:49
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayBatchIterator class is used as a wrapper around a Camera to easily
 *   iterate over Rays. In contrast to the normal RayIterator, does this
 *   iterator return a bunch of rays (a RayBatch) to facilitate massive
 *   parallelism (a.k.a., the GPU) over a portion of all available rays, to
 *   reduce memory usage.
**/

#include "GPUTools.hpp"

#include "RayBatchIterator.hpp"

using namespace std;
using namespace RayTracer;


/***** RAY BATCH ITERATOR *****/
HOST_DEVICE RayBatch::iterator::iterator(RayBatch* data) :
    data(data),
    pos(0),
    max(this->data->n_rays)
{}

HOST_DEVICE RayBatch::iterator::iterator(RayBatch* data, size_t pos) :
    data(data),
    pos(pos),
    max(this->data->n_rays)
{
    if (this->pos > this->max) {
        this->pos = this->max;
    }
}



HOST_DEVICE RayBatch::iterator& RayBatch::iterator::operator++() {
    // Increment this pos, return
    this->pos++;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}

HOST_DEVICE RayBatch::iterator RayBatch::iterator::operator+(size_t n) const {
    size_t new_pos = this->pos + n;
    if (new_pos > this->max) { new_pos = this->max; }
    return RayBatch::iterator(this->data, new_pos);
}

HOST_DEVICE RayBatch::iterator& RayBatch::iterator::operator+=(size_t n) {
    // Increment this pos, return
    this->pos += n;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}





/***** RAY BATCH CONST ITERATOR *****/
HOST_DEVICE RayBatch::const_iterator::const_iterator(const RayBatch* data) :
    data(data),
    pos(0),
    max(this->data->n_rays)
{}

HOST_DEVICE RayBatch::const_iterator::const_iterator(const RayBatch* data, size_t pos) :
    data(data),
    pos(pos),
    max(this->data->n_rays)
{
    if (this->pos > this->max) {
        this->pos = this->max;
    }
}



HOST_DEVICE RayBatch::const_iterator& RayBatch::const_iterator::operator++() {
    // Increment this pos, return
    this->pos++;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}

HOST_DEVICE RayBatch::const_iterator RayBatch::const_iterator::operator+(size_t n) const {
    size_t new_pos = this->pos + n;
    if (new_pos > this->max) { new_pos = this->max; }
    return RayBatch::const_iterator(this->data, new_pos);
}

HOST_DEVICE RayBatch::const_iterator& RayBatch::const_iterator::operator+=(size_t n) {
    // Increment this pos, return
    this->pos += n;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}





/***** RAY BATCH *****/
HOST_DEVICE RayBatch::RayBatch(const Camera& camera, size_t n_rays, size_t start, size_t stop) :
    n_rays(stop - start),
    start(start),
    stop(stop),
    width(camera.frame_width),
    height(camera.frame_height),
    rays_per_pixel(n_rays)
{
    // Allocate the Rays list
    this->rays = new Ray[this->n_rays];

    // Populate it with stuff from the camera
    for (size_t i = start; i < stop; i++) {
        // Decode into x & y
        size_t r = i / this->rays_per_pixel;
        size_t x = r % this->width;
        size_t y = r / this->width;

        // Cast the ray
        this->rays[i] = camera.cast(x, y);
    }
}

HOST_DEVICE RayBatch::RayBatch(const RayBatch& other) :
    n_rays(other.n_rays),
    start(other.start),
    stop(other.stop),
    width(other.width),
    height(other.height),
    rays_per_pixel(other.rays_per_pixel)
{
    this->rays = new Ray[this->n_rays];
    
    // Copy all rays
    for (size_t i = 0; i < this->n_rays; i++) {
        this->rays[i] = other.rays[i];
    }
}

HOST_DEVICE RayBatch::RayBatch(RayBatch&& other) :
    n_rays(other.n_rays),
    start(other.start),
    stop(other.stop),
    width(other.width),
    height(other.height),
    rays_per_pixel(other.rays_per_pixel)
{
    // Steal the rays list
    this->rays = other.rays;
    other.rays = nullptr;
}

HOST_DEVICE RayBatch::~RayBatch() {
    // Only deallocate when != nullptr, to accomodate for being robbed
    if (this->rays != nullptr) {
        delete[] this->rays;
    }
}



#ifdef CUDA
RayBatch* RayBatch::GPU_create(const Camera& camera, size_t n_rays, size_t start, size_t stop, void* ptr) {
    CUDA_DEBUG("RayBatch GPU-constructor");

    // Create a template to copy from
    RayBatch temp(camera, n_rays, start, stop);

    // Always allocate space for the data & copy
    Ray* data;
    cudaMalloc((void**) &data, sizeof(Ray) * temp.n_rays);
    CUDA_ASSERT("Could not allocate additional data");
    cudaMemcpy((void*) data, (void*) temp.rays, sizeof(Ray) * temp.n_rays, cudaMemcpyHostToDevice);
    CUDA_ASSERT("Could not copy additional data to device");

    // Deallocate the struct ptr & replace with this one
    delete[] temp.rays;
    temp.rays = data;

    // Only allocate for the struct itself if needed
    RayBatch* ptr_gpu = (RayBatch*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &data, sizeof(RayBatch));
        CUDA_MALLOC_ASSERT();
    }

    // Copy both it to the GPU
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(RayBatch), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Replace the pointer in the struct with nullptr as we're done with that and don't want it to try and deallocate GPU-stuff.
    temp.rays = nullptr;

    // Return the pointer
    return ptr_gpu;
}

RayBatch* RayBatch::GPU_create(const RayBatch& other, void* ptr) {
    CUDA_DEBUG("RayBatch GPU-copy constructor");

    // Create a template to copy from
    RayBatch temp(other);

    // Always allocate space for the data & copy
    Ray* data;
    cudaMalloc((void**) &data, sizeof(Ray) * temp.n_rays);
    CUDA_ASSERT("Could not allocate additional data");
    cudaMemcpy((void*) data, (void*) temp.rays, sizeof(Ray) * temp.n_rays, cudaMemcpyHostToDevice);
    CUDA_ASSERT("Could not copy additional data to device");

    // Deallocate the struct ptr & replace with this one
    delete[] temp.rays;
    temp.rays = data;

    // Only allocate for the struct itself if needed
    RayBatch* ptr_gpu = (RayBatch*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(RayBatch));
        CUDA_MALLOC_ASSERT();
    }

    // Copy both it to the GPU
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(RayBatch), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Replace the pointer in the struct with nullptr as we're done with that and don't want it to try and deallocate GPU-stuff.
    temp.rays = nullptr;

    // Return the pointer
    return ptr_gpu;
}

RayBatch RayBatch::GPU_copy(RayBatch* ptr_gpu) {
    CUDA_DEBUG("RayBatch GPU-copy");
    
    // Copy the given RayBatch object into a buffer
    char buffer[sizeof(RayBatch)];
    cudaMemcpy((void*) buffer, (void*) ptr_gpu, sizeof(RayBatch), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Read the buffer as RayBatch
    RayBatch& result = *((RayBatch*) buffer);

    // Allocate the appropriate space on the CPU
    Ray* rays = new Ray[result.n_rays];

    // Copy the GPU-side rays list into it
    cudaMemcpy((void*) rays, (void*) result.rays, sizeof(Ray) * result.n_rays, cudaMemcpyDeviceToHost);
    CUDA_ASSERT("Could not copy additional data to host");

    // Link it to the resulting batch
    result.rays = rays;

    // Done, return as rvalue to avoid copying the rays list again
    return std::move(result);
}

void RayBatch::GPU_free(RayBatch* ptr_gpu) {
    CUDA_DEBUG("RayBatch GPU-destructor");
    
    // Copy the given RayBatch object into a buffer
    char buffer[sizeof(RayBatch)];
    cudaMemcpy((void*) buffer, (void*) ptr_gpu, sizeof(RayBatch), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Read the buffer as RayBatch
    RayBatch& result = *((RayBatch*) buffer);

    // Deallocate the Rays list
    cudaFree(result.rays);
    CUDA_ASSERT("Could not free additional data");
    // Deallocate the RayBatch struct itself
    cudaFree(ptr_gpu);
    CUDA_FREE_ASSERT();
}

#endif



HOST_DEVICE RayBatch& RayBatch::operator=(RayBatch&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(RayBatch& rb1, RayBatch& rb2) {
    // Swap the members
    swap(rb1.rays, rb2.rays);
    swap(rb1.n_rays, rb2.n_rays);
    swap(rb1.start, rb2.start);
    swap(rb1.stop, rb2.stop);
    swap(rb1.width, rb2.width);
    swap(rb1.height, rb2.height);
    swap(rb1.rays_per_pixel, rb2.rays_per_pixel);
}





/***** RAYBATCH ITERATOR CONST ITERATOR *****/
RayBatchIterator::const_iterator::const_iterator(const RayBatchIterator* data) :
    data(data),
    pos(0),
    max(this->data->camera.frame_width * this->data->camera.frame_height * this->data->n_rays / this->data->batch_size)
{}

RayBatchIterator::const_iterator::const_iterator(const RayBatchIterator* data, size_t pos) :
    data(data),
    pos(pos),
    max(this->data->camera.frame_width * this->data->camera.frame_height * this->data->n_rays / this->data->batch_size)
{
    // Make sure it never surpasses the end
    if (this->pos > this->max) {
        this->pos = this->max;
    }
}



RayBatchIterator::const_iterator& RayBatchIterator::const_iterator::operator++() {
    // Increment this pos bounded, return
    this->pos++;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}

RayBatchIterator::const_iterator RayBatchIterator::const_iterator::operator+(size_t n) const {
    size_t new_pos = this->pos + n;
    if (new_pos > this->max) { new_pos = this->max; }
    return RayBatchIterator::const_iterator(this->data, new_pos);
}

RayBatchIterator::const_iterator& RayBatchIterator::const_iterator::operator+=(size_t n) {
    // Increment this pos, return
    this->pos += n;
    if (this->pos > this->max) { this->pos = this->max; }
    return *this;
}





/***** RAYBATCH ITERATOR *****/
RayBatchIterator::RayBatchIterator(Camera& camera, size_t n_rays, size_t batch_size) :
    camera(camera),
    n_rays(n_rays),
    batch_size(batch_size)
{}

RayBatchIterator::RayBatchIterator(const RayBatchIterator& other) :
    camera(other.camera),
    n_rays(other.n_rays),
    batch_size(other.batch_size)
{}

RayBatchIterator::RayBatchIterator(RayBatchIterator&& other) :
    camera(other.camera),
    n_rays(other.n_rays),
    batch_size(other.batch_size)
{}



RayBatch RayBatchIterator::operator[](size_t n) const {
    size_t max = this->camera.frame_width * this->camera.frame_height * this->n_rays;
    
    // Get n1, or otherwise the n pointing after the last element if n is larger
    size_t n1 = n * this->batch_size;
    if (n1 > max) {
        n1 = max;
    }

    // Get n2, and bound it as well
    size_t n2 = n1 + this->batch_size;
    if (n2 > max) {
        n2 = max;
    }

    // Return a RayBatch with appropriate dimensions
    return RayBatch(this->camera, this->n_rays, n1, n2);
}



RayBatchIterator& RayBatchIterator::operator=(RayBatchIterator&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

void RayTracer::swap(RayBatchIterator& ri1, RayBatchIterator& ri2) {
    using std::swap;

    // Swap the camera
    swap(ri1.camera, ri2.camera);
    // Swap the number of rays
    swap(ri1.n_rays, ri2.n_rays);
    // Swap the batch size
    swap(ri1.batch_size, ri2.batch_size);
}

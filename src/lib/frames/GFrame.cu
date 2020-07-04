/* GFRAME.cu
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 3:13:03 PM
 * Last edited:
 *   7/4/2020, 4:18:18 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The GFrame-class is the GPU-side counterpart of the Frame class. It
 *   specializes in copying the data to and from the GPU and in accessing
 *   operations on the GPU - but not in writing to a PNG, as the GPU has no
 *   file access anyway.
**/

#include "GFrame.hpp"

using namespace std;
using namespace RayTracer;


/***** GPIXEL CLASS *****/

__device__ GPixel::GPixel(const Coordinate& pos, double* const data) :
    data(data),
    pos(pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

__device__ GPixel::GPixel(const GPixel& other) :
    data(other.data),
    pos(other.pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

__device__ GPixel::GPixel(GPixel&& other) :
    data(other.data),
    pos(other.pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}



__device__ double GPixel::operator[](const size_t i) const {
    // Check if within bounds
    if (i > 2) { printf("ERROR: double GPixel::operator[](const size_t i) const: Index %lu is out of range for GPixel with size 3.", i); }

    // Return
    return this->data[i];
}

__device__ double& GPixel::operator[](const size_t i) {
    // Check if within bounds
    if (i > 2) { printf("ERROR: double GPixel::operator[](const size_t i) const: Index %lu is out of range for GPixel with size 3.", i); }

    // Return
    return this->data[i];
}





/***** GFRAME CLASS *****/

__device__ GFrame::GFrame(cudaPitchedPtr ptr) :
    width(ptr.xsize),
    height(ptr.ysize)
{
    this->data = ptr.ptr;
    this->pitch = ptr.pitch;
}



__device__ const GPixel GFrame::operator[](const Coordinate& index) const {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: const GPixel GFrame::operator[](const Coordinate& index) const: X-axis index %lu is out fo range for GFrame of %lu by %lu.", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: const GPixel GFrame::operator[](const Coordinate& index) const: Y-axis index %lu is out fo range for GFrame of %lu by %lu.", index.y, this->width, this->height);
    }

    // Return the correct index GPixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return GPixel(index, ptr);
}

__device__ GPixel GFrame::operator[](const Coordinate& index) {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: GPixel GFrame::operator[](const Coordinate& index): X-axis index %lu is out fo range for GFrame of %lu by %lu.", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: GPixel GFrame::operator[](const Coordinate& index): Y-axis index %lu is out fo range for GFrame of %lu by %lu.", index.y, this->width, this->height);
    }

    // Return the correct index GPixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return GPixel(index, ptr);
}



__host__ cudaPitchedPtr GFrame::toGPU(const Frame& frame) {
    size_t w = sizeof(double) * frame.width * 3;
    size_t h = frame.height;

    // Allocate enough space on the GPU
    cudaPitchedPtr ptr;
    cudaMallocPitch(&ptr.ptr, &ptr.pitch, w, h);

    // Copy the stuff to the GPU
    cudaMemcpy2D(ptr.ptr, ptr.pitch, frame.data, w, w, h, cudaMemcpyHostToDevice);

    // Return the pointer & pitch
    ptr.xsize = frame.width;
    ptr.ysize = frame.height;
    return ptr;
}

__host__ Frame GFrame::fromGPU(cudaPitchedPtr ptr) {
    // Create a buffer for the frame
    size_t w = sizeof(double) * ptr.xsize * 3;
    size_t h = ptr.ysize;
    double* data = new double[ptr.xsize * ptr.ysize * 3];

    // Copy all data to the Frame
    cudaMemcpy2D((void*) data, w, ptr.ptr, ptr.pitch, w, h, cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(ptr.ptr);

    // Return a new Frame object with given data
    return Frame(ptr.xsize, ptr.ysize, data);
}



__device__ GFrame& GFrame::operator=(GFrame other) {
    // Only do stuff if not ourselves
    if (this != &other) {
        // Check if the size is correct
        if (this->width != other.width || this->height != other.height) {
            printf("ERROR: GFrame& GFrame::operator=(GFrame other): Cannot copy the value of a GFrame with different dimensions.");
        }

        // Swap 'em
        swap(*this, other);
    }

    return *this;
}

__device__ GFrame& GFrame::operator=(GFrame&& other) {
    // Only do stuff if not ourselves
    if (this != &other) {
        // Check if the size is correct
        if (this->width != other.width || this->height != other.height) {
            printf("ERROR: GFrame& GFrame::operator=(GFrame&& other): Cannot copy the value of a GFrame with different dimensions.");
        }

        // Swap 'em
        swap(*this, other);
    }

    return *this;
}

__device__ void RayTracer::swap(GFrame& f1, GFrame& f2) {
    using std::swap;

    // Swap the data pointers
    void* tdata = f1.data;
    f1.data = f2.data;
    f2.data = tdata;
}





/***** ITERATOR *****/

__device__ GFrame::iterator::iterator(GFrame* frame) {
    this->pos.x = 0;
    this->pos.y = 0;
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;
}

__device__ GFrame::iterator::iterator(GFrame* frame, const Coordinate& pos) {
    this->pos.x = pos.x;
    this->pos.y = pos.y; 
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;  
}



__device__ GFrame::iterator& GFrame::iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

__device__ GFrame::iterator GFrame::iterator::operator+(size_t n) const {
    Coordinate n_pos({this->pos.x + n, this->pos.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return GFrame::iterator(this->data, n_pos);
}

__device__ GFrame::iterator GFrame::iterator::operator+(const Coordinate& n) const {
    Coordinate n_pos({this->pos.x + n.x, this->pos.y + n.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return GFrame::iterator(this->data, n_pos);
}

__device__ GFrame::iterator& GFrame::iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

__device__ GFrame::iterator& GFrame::iterator::operator+=(const Coordinate& n) {
    this->pos.x += n.x;
    this->pos.y += n.y;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}





/***** CONSTANT ITERATOR *****/

__device__ GFrame::const_iterator::const_iterator(const GFrame* frame) {
    this->pos.x = 0;
    this->pos.y = 0;
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;
}

__device__ GFrame::const_iterator::const_iterator(const GFrame* frame, const Coordinate& pos) {
    this->pos.x = pos.x;
    this->pos.y = pos.y; 
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;  
}



__device__ GFrame::const_iterator& GFrame::const_iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

__device__ GFrame::const_iterator GFrame::const_iterator::operator+(size_t n) const {
    Coordinate n_pos({this->pos.x + n, this->pos.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return GFrame::const_iterator(this->data, n_pos);
}

__device__ GFrame::const_iterator GFrame::const_iterator::operator+(const Coordinate& n) const {
    Coordinate n_pos({this->pos.x + n.x, this->pos.y + n.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return GFrame::const_iterator(this->data, n_pos);
}

__device__ GFrame::const_iterator& GFrame::const_iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

__device__ GFrame::const_iterator& GFrame::const_iterator::operator+=(const Coordinate& n) {
    this->pos.x += n.x;
    this->pos.y += n.y;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

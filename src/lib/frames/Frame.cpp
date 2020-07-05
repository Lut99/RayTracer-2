/* FRAME.cpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:00 PM
 * Last edited:
 *   05/07/2020, 17:21:43
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Frame class is a container for the resulting pixels which can be
 *   relatively easily written to PNG. In fact, the writing to a PNG-file
 *   is also supported.
**/

#include <stdexcept>
#include <iostream>
#include <fstream>
#include <vector>

#include "Frame.hpp"
#include "LodePNG.hpp"

using namespace std;
using namespace RayTracer;


/***** PIXEL CLASS *****/

HOST_DEVICE Pixel::Pixel(const Coordinate& pos, double* const data) :
    data(data),
    pos(pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

HOST_DEVICE Pixel::Pixel(const Pixel& other) :
    data(other.data),
    pos(other.pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

HOST_DEVICE Pixel::Pixel(Pixel&& other) :
    data(other.data),
    pos(other.pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}



HOST_DEVICE double Pixel::operator[](const size_t i) const {
    // Check if within bounds
    if (i > 2) { printf("ERROR: double Pixel::operator[](const size_t i) const: Index %lu is out of range for Pixel with size 3.\n", i); }

    // Return
    return this->data[i];
}

HOST_DEVICE double& Pixel::operator[](const size_t i) {
    // Check if within bounds
    if (i > 2) { printf("ERROR: double Pixel::operator[](const size_t i) const: Index %lu is out of range for Pixel with size 3.\n", i); }

    // Return
    return this->data[i];
}





/***** FRAME CLASS *****/

Frame::Frame(size_t width, size_t height) :
    width(width),
    height(height)
{
    this->data = (void*) new double[width * height * 3];
    this->pitch = sizeof(double) * width * 3;
    this->is_external = false;
}

Frame::Frame(size_t width, size_t height, void* data) :
    width(width),
    height(height)
{
    this->data = data;
    this->pitch = sizeof(double) * width * 3;
    this->is_external = false;
}

#ifdef CUDA
__device__ Frame::Frame(const FramePtr& ptr) :
    width(ptr.width),
    height(ptr.height)
{
    this->data = ptr.data;
    this->pitch = ptr.pitch;
    this->is_external = true;
}
#endif

HOST_DEVICE Frame::Frame(const Frame& other) :
    width(other.width),
    height(other.height)
{
    // Allocate a new buffer
    this->data = (void*) new double[this->width * this->height * 3];
    this->pitch = sizeof(double) * this->width * 3;
    this->is_external = false;

    // Copy the data
    double* dest = (double*) this->data;
    double* source = (double*) other.data;
    for (size_t i = 0; i < this->width * this->height * 3; i++) {
        dest[i] = source[i];
    }
}

HOST_DEVICE Frame::Frame(Frame&& other) :
    width(other.width),
    height(other.height)
{
    // Steal the data
    this->data = other.data;
    this->pitch = other.pitch;
    this->is_external = other.is_external;
    other.is_external = true;
}

HOST_DEVICE Frame::~Frame() {
    // Only delete if not local (in case we've been robbed)
    if (!this->is_external) {
        delete[] ((double*) this->data);
    }
}



HOST_DEVICE const Pixel Frame::operator[](const Coordinate& index) const {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: const Pixel Frame::operator[](const Coordinate& index) const: X-axis index %lu is out of range for Frame of %lu by %lu.\n", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: const Pixel Frame::operator[](const Coordinate& index) const: Y-axis index %lu is out of range for Frame of %lu by %lu.\n", index.y, this->width, this->height);
    }

    // Return the correct index Pixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return Pixel(index, ptr);
}

HOST_DEVICE Pixel Frame::operator[](const Coordinate& index) {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: Pixel Frame::operator[](const Coordinate& index): X-axis index %lu is out of range for Frame of %lu by %lu.\n", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: Pixel Frame::operator[](const Coordinate& index): Y-axis index %lu is out of range for Frame of %lu by %lu.\n", index.y, this->width, this->height);
    }

    // Return the correct index Pixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return Pixel(index, ptr);
}



#ifdef CUDA
FramePtr Frame::toGPU(void* data) const {
    size_t w = sizeof(double) * this->width * 3;
    size_t h = this->height;

    // Allocate space on the GPU if needed
    FramePtr ptr;
    if (data == nullptr) {
        cudaMallocPitch(&ptr.data, &ptr.pitch, w, h);
    } else {
        ptr.data = data;
        ptr.pitch = w;
    }

    // Copy the stuff to the GPU
    cudaMemcpy2D(ptr.data, ptr.pitch, this->data, w, w, h, cudaMemcpyHostToDevice);

    // Return the pointer & pitch
    ptr.width = this->width;
    ptr.height = this->height;
    return ptr;
}

Frame Frame::fromGPU(const FramePtr& ptr) {
    // Create a buffer for the frame
    size_t w = sizeof(double) * ptr.width * 3;
    size_t h = ptr.height;
    void* data = (void*) new double[ptr.width * ptr.height * 3];

    // Copy all data to the Frame
    cudaMemcpy2D(data, w, ptr.data, ptr.pitch, w, h, cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(ptr.data);

    // Return a new Frame object with given data
    return Frame(ptr.width, ptr.height, data);
}
#endif



void Frame::toPNG(const string& path) const {
    // Generate a 0-255, four channel vector
    vector<unsigned char> raw_image;
    raw_image.resize(this->width * this->height * 4);
    for (Pixel p : *this) {
        size_t x = p.x();
        size_t y = p.y();
        raw_image[4 * (y * this->width + x) + 0] = (char) (255.0 * p.r);
        raw_image[4 * (y * this->width + x) + 1] = (char) (255.0 * p.g);
        raw_image[4 * (y * this->width + x) + 2] = (char) (255.0 * p.b);
        raw_image[4 * (y * this->width + x) + 3] = 255;
    }

    // Write that to the output file using LodePNG
    unsigned result = lodepng::encode(path.c_str(), raw_image, this->width, this->height);
    if (result != 0) {
        cerr << "ERROR: void Frame::toPNG(const string& path) const: Could not write to PNG file: " << lodepng_error_text(result) << endl;
        exit(1);
    }
}



HOST_DEVICE Frame& Frame::operator=(Frame other) {
    // Check if the size is correct
    if (this->width != other.width || this->height != other.height) {
        printf("ERROR: Frame& Frame::operator=(Frame other): Cannot copy the value of a Frame with different dimensions.");
    } else {
        // Swap 'em
        swap(*this, other);
    }

    return *this;
}

HOST_DEVICE Frame& Frame::operator=(Frame&& other) {
    // Only do stuff if not ourselves
    if (this != &other) {
        // Check if the size is correct
        if (this->width != other.width || this->height != other.height) {
            printf("ERROR: Frame& Frame::operator=(Frame&& other): Cannot copy the value of a Frame with different dimensions.");
        } else {
            // Swap 'em
            swap(*this, other);
        }
    }

    return *this;
}

HOST_DEVICE void RayTracer::swap(Frame& f1, Frame& f2) {
    // Swap the data pointers
    void* tdata = f1.data;
    f1.data = f2.data;
    f2.data = tdata;
}





/***** ITERATOR *****/

HOST_DEVICE Frame::iterator::iterator(Frame* frame) {
    this->pos.x = 0;
    this->pos.y = 0;
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;
}

HOST_DEVICE Frame::iterator::iterator(Frame* frame, const Coordinate& pos) {
    this->pos.x = pos.x;
    this->pos.y = pos.y; 
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;  
}



HOST_DEVICE Frame::iterator& Frame::iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

HOST_DEVICE Frame::iterator Frame::iterator::operator+(size_t n) const {
    Coordinate n_pos({this->pos.x + n, this->pos.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return Frame::iterator(this->data, n_pos);
}

HOST_DEVICE Frame::iterator Frame::iterator::operator+(const Coordinate& n) const {
    Coordinate n_pos({this->pos.x + n.x, this->pos.y + n.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return Frame::iterator(this->data, n_pos);
}

HOST_DEVICE Frame::iterator& Frame::iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

HOST_DEVICE Frame::iterator& Frame::iterator::operator+=(const Coordinate& n) {
    this->pos.x += n.x;
    this->pos.y += n.y;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}





/***** CONSTANT ITERATOR *****/

HOST_DEVICE Frame::const_iterator::const_iterator(const Frame* frame) {
    this->pos.x = 0;
    this->pos.y = 0;
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;
}

HOST_DEVICE Frame::const_iterator::const_iterator(const Frame* frame, const Coordinate& pos) {
    this->pos.x = pos.x;
    this->pos.y = pos.y; 
    this->width = frame->width;
    this->height = frame->height;
    this->data = frame;  
}



HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

HOST_DEVICE Frame::const_iterator Frame::const_iterator::operator+(size_t n) const {
    Coordinate n_pos({this->pos.x + n, this->pos.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return Frame::const_iterator(this->data, n_pos);
}

HOST_DEVICE Frame::const_iterator Frame::const_iterator::operator+(const Coordinate& n) const {
    Coordinate n_pos({this->pos.x + n.x, this->pos.y + n.y});
    n_pos.y += n_pos.x / this->width;
    n_pos.x = n_pos.x % this->width;
    return Frame::const_iterator(this->data, n_pos);
}

HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator+=(const Coordinate& n) {
    this->pos.x += n.x;
    this->pos.y += n.y;
    this->pos.y += this->pos.x / this->width;
    this->pos.x = this->pos.x % this->width;
    return *this;
}

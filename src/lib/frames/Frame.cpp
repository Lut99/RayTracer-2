/* FRAME.cpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:00 PM
 * Last edited:
 *   13/07/2020, 14:36:55
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


/***** HELPER FUNCTIONS *****/

/* Returns two of either references, based on the boolean. */
double& get_reference(double& d1, double& d2, bool should_be_two) {
    if (should_be_two) {
        return d2;
    }
    return d1;
}





/***** PIXEL CLASS *****/

HOST_DEVICE Pixel::Pixel(const Point2& pos, double* const data) :
    data(data),
    is_external(true),
    frame_pos(pos),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

HOST_DEVICE Pixel::Pixel(double r, double g, double b) :
    is_external(false),
    frame_pos(0, 0),
    local_r(r),
    local_g(g),
    local_b(b),
    r(this->local_r),
    g(this->local_g),
    b(this->local_b)
{}

HOST_DEVICE Pixel::Pixel(const Pixel& other) :
    is_external(false),
    frame_pos(0, 0),
    local_r(other.r),
    local_g(other.g),
    local_b(other.b),
    r(this->local_r),
    g(this->local_g),
    b(this->local_b)
{}

HOST_DEVICE Pixel::Pixel(Pixel&& other) :
    is_external(false),
    frame_pos(0, 0),
    local_r(other.r),
    local_g(other.g),
    local_b(other.b),
    r(this->local_r),
    g(this->local_g),
    b(this->local_b)
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



HOST_DEVICE Pixel& Pixel::operator=(Pixel&& other) {
    // Swap the two if and only if they aren't the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(Pixel& p1, Pixel& p2) {
    // Swap the three values, regardless of their underlying location
    double t = p1.r;
    p1.r = p2.r;
    p2.r = t;

    t = p1.g;
    p1.g = p2.g;
    p2.g = t;

    t = p1.b;
    p1.b = p2.b;
    p2.b = t;

    // Also swap the Coordinates
    swap(p1.frame_pos, p2.frame_pos);
}



std::ostream& RayTracer::operator<<(std::ostream& os, const Pixel& pixel) {
    os << "(" << pixel.r << ", " << pixel.g << ", " << pixel.b << ")";
    return os;
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

Frame::Frame(size_t width, size_t height, void* data, size_t pitch) :
    width(width),
    height(height)
{
    this->data = data;
    this->pitch = pitch;
    this->is_external = true;
}

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



#ifdef CUDA
#include <iostream>
Frame* Frame::GPU_create(size_t width, size_t height, void* ptr) {
    CUDA_DEBUG("Frame GPU-constructor");

    // First, allocate the GPU-side data array
    void* data;
    size_t pitch;
    cudaMallocPitch(&data, &pitch, sizeof(double) * width * 3, height);
    CUDA_ASSERT("Could not allocate additional data");

    // Then, create an empty cpu-side Frame with this pointer as target
    Frame temp(width, height, data, pitch);

    // If needed, allocate space for the Frame on the GPU.
    Frame* ptr_gpu = (Frame*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Frame));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the struct itself
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Frame), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Frame* Frame::GPU_create(const Frame& other, void* ptr) {
    CUDA_DEBUG("Frame GPU-copy constructor");

    // First, allocate the GPU-side data array
    void* data;
    size_t pitch;
    cudaMallocPitch(&data, &pitch, sizeof(double) * other.width * 3, other.height);
    CUDA_ASSERT("Could not allocate additional data");

    // Then, copy the CPU-side data to the GPU
    cudaMemcpy2D(data, pitch, other.data, other.pitch, sizeof(double) * other.width * 3, other.height, cudaMemcpyHostToDevice);
    CUDA_ASSERT("Could not copy additional data to device");

    // Then, create an empty frame with the given frame's properties
    Frame temp(other.width, other.height, data, pitch);

    // If needed, allocate space for the Frame on the GPU.
    Frame* ptr_gpu = (Frame*) ptr;
    if (ptr_gpu == nullptr) {
        cudaMalloc((void**) &ptr_gpu, sizeof(Frame));
        CUDA_MALLOC_ASSERT();
    }

    // Copy the struct itself
    cudaMemcpy((void*) ptr_gpu, (void*) &temp, sizeof(Frame), cudaMemcpyHostToDevice);
    CUDA_COPYTO_ASSERT();

    // Return the pointer
    return ptr_gpu;
}

Frame Frame::GPU_copy(Frame* ptr_gpu) {
    CUDA_DEBUG("Frame GPU-copy");
    
    // Create a CPU-side buffer for the data
    char buffer[sizeof(Frame)];

    // Copy the GPU-side frame data into it
    cudaMemcpy((void*) buffer, (void*) ptr_gpu, sizeof(Frame), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();

    // Extract the Frame
    Frame& result = *((Frame*) buffer);

    // Copy the data into that frame
    void* data = (void*) new double[result.width * result.height * 3];
    size_t width = sizeof(double) * result.width * 3;
    cudaMemcpy2D(data, width, result.data, result.pitch, width, result.height, cudaMemcpyDeviceToHost);
    CUDA_ASSERT("Could not copy additional data to host");

    // Swap the pointers in the given struct
    result.data = data;
    result.pitch = width;
    result.is_external = false;

    // Return
    return result;
}

void Frame::GPU_free(Frame* ptr_gpu) {
    CUDA_DEBUG("Frame GPU-destructor");
    
    // First, fetch a copy of the Frame to know the data pointer
    char buffer[sizeof(Frame)];
    cudaMemcpy((void*) buffer, (void*) ptr_gpu, sizeof(Frame), cudaMemcpyDeviceToHost);
    CUDA_COPYFROM_ASSERT();
    Frame& result = *((Frame*) buffer);

    // Then, clear both
    cudaFree(result.data);
    CUDA_ASSERT("Could not free additional data");
    cudaFree(ptr_gpu);
    CUDA_FREE_ASSERT();
}
#endif



HOST_DEVICE const Pixel Frame::operator[](const Point2& index) const {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: const Pixel Frame::operator[](const Point2& index) const: X-axis index %lu is out of range for Frame of %lu by %lu.\n", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: const Pixel Frame::operator[](const Point2& index) const: Y-axis index %lu is out of range for Frame of %lu by %lu.\n", index.y, this->width, this->height);
    }

    // Return the correct index Pixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return Pixel(index, ptr);
}

HOST_DEVICE Pixel Frame::operator[](const Point2& index) {
    // Check if within bounds
    if (index.x >= this->width) {
        printf("ERROR: Pixel Frame::operator[](const Point2& index): X-axis index %lu is out of range for Frame of %lu by %lu.\n", index.x, this->width, this->height);
    }
    if (index.y >= this->height) {
        printf("ERROR: Pixel Frame::operator[](const Point2& index): Y-axis index %lu is out of range for Frame of %lu by %lu.\n", index.y, this->width, this->height);
    }

    // Return the correct index Pixel
    double* ptr = (double*) ((char*) this->data + index.y * this->pitch) + index.x * 3;
    return Pixel(index, ptr);
}



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



HOST_DEVICE Frame& Frame::operator=(const Frame& other) {
    // Check if the size is correct
    if (this->width != other.width || this->height != other.height) {
        printf("ERROR: Frame& Frame::operator=(Frame other): Cannot copy the value of a Frame with different dimensions.");
        return *this;
    }

    return *this = Frame(other);
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

HOST_DEVICE Frame::iterator::iterator(Frame* frame) :
    data(frame),
    pos(0, 0),
    max(0, this->data->height)
{}

HOST_DEVICE Frame::iterator::iterator(Frame* frame, const Point2& pos) :
    data(frame),
    pos(pos),
    max(0, this->data->height)
{
    if (this->pos > this->max) {
        this->pos = this->max;
    }
}



HOST_DEVICE Frame::iterator& Frame::iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}

HOST_DEVICE Frame::iterator Frame::iterator::operator+(size_t n) const {
    Point2 new_pos(this->pos.x + n, this->pos.y);
    new_pos.y += new_pos.x / this->data->width;
    new_pos.x = new_pos.x % this->data->width;
    if (new_pos > this->max) {
        new_pos = this->max;
    }
    return Frame::iterator(this->data, new_pos);
}

HOST_DEVICE Frame::iterator Frame::iterator::operator+(const Point2& n) const {
    Point2 new_pos = this->pos + n;
    new_pos.y += new_pos.x / this->data->width;
    new_pos.x = new_pos.x % this->data->width;
    if (new_pos > this->max) {
        new_pos = this->max;
    }
    return Frame::iterator(this->data, new_pos);
}

HOST_DEVICE Frame::iterator& Frame::iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}

HOST_DEVICE Frame::iterator& Frame::iterator::operator+=(const Point2& n) {
    this->pos += n;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}





/***** CONSTANT ITERATOR *****/

HOST_DEVICE Frame::const_iterator::const_iterator(const Frame* frame) :
    data(frame),
    pos(0, 0),
    max(0, this->data->height)
{}

HOST_DEVICE Frame::const_iterator::const_iterator(const Frame* frame, const Point2& pos) :
    data(frame),
    pos(pos),
    max(0, this->data->height)
{
    // Make sure pos is within bounds
    if (this->pos > this->max) {
        this->pos = this->max;
    }
}



HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator++() {
    this->pos.x++;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}

HOST_DEVICE Frame::const_iterator Frame::const_iterator::operator+(size_t n) const {
    Point2 new_pos(this->pos.x + n, this->pos.y);
    new_pos.y += new_pos.x / this->data->width;
    new_pos.x = new_pos.x % this->data->width;
    if (new_pos > this->max) {
        new_pos = this->max;
    }
    return Frame::const_iterator(this->data, new_pos);
}

HOST_DEVICE Frame::const_iterator Frame::const_iterator::operator+(const Point2& n) const {
    Point2 new_pos = this->pos + n;
    new_pos.y += new_pos.x / this->data->width;
    new_pos.x = new_pos.x % this->data->width;
    if (new_pos > this->max) {
        new_pos = this->max;
    }
    return Frame::const_iterator(this->data, new_pos);
}

HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator+=(size_t n) {
    this->pos.x += n;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}

HOST_DEVICE Frame::const_iterator& Frame::const_iterator::operator+=(const Point2& n) {
    this->pos += n;
    this->pos.y += this->pos.x / this->data->width;
    this->pos.x = this->pos.x % this->data->width;
    if (this->pos > this->max) {
        this->pos = this->max;
    }
    return *this;
}

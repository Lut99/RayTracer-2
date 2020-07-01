/* FRAME.cpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:00 PM
 * Last edited:
 *   7/1/2020, 6:04:03 PM
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

Pixel::Pixel(double* const data) :
    data(data),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

Pixel::Pixel(const Pixel& other) :
    data(other.data),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}

Pixel::Pixel(Pixel&& other) :
    data(other.data),
    r(this->data[0]),
    g(this->data[1]),
    b(this->data[2])
{}



double Pixel::operator[](const size_t i) const {
    // Check if within bounds
    if (i > 2) { throw std::out_of_range("ERROR: double Pixel::operator[](const size_t i) const: Index " + to_string(i) + " is out of range for Pixel with size 3."); }

    // Return
    return this->data[i];
}

double& Pixel::operator[](const size_t i) {
    // Check if within bounds
    if (i > 2) { throw std::out_of_range("ERROR: double Pixel::operator[](const size_t i) const: Index " + to_string(i) + " is out of range for Pixel with size 3."); }

    // Return
    return this->data[i];
}



Pixel& Pixel::operator=(Pixel other) {
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

Pixel& Pixel::operator=(Pixel&& other) {
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

void RayTracer::swap(Pixel& p1, Pixel& p2) {
    // Swap the channels, as this make sure the values for the data is swapped as well
    double t = p1.r;
    p1.r = p2.r;
    p2.r = t;

    t = p1.g;
    p1.g = p2.g;
    p2.g = t;

    t = p1.b;
    p1.b = p2.b;
    p2.b = t;
}




/***** FRAME CLASS *****/

Frame::Frame(size_t width, size_t height) :
    width(width),
    height(height)
{
    this->data = new double[width * height * 3];
}

Frame::Frame(const Frame& other) :
    width(other.width),
    height(other.height)
{
    // Copy the data
    this->data = new double[width * height * 3];
    for (size_t i = 0; i < width * height * 3; i++) {
        this->data[i] = other.data[i];
    }
}

Frame::Frame(Frame&& other) :
    width(other.width),
    height(other.height)
{
    // Steal the data
    this->data = other.data;
    other.data = nullptr;
}

Frame::~Frame() {
    // Only delete is not null (in case we've been robbed)
    if (this->data != nullptr) {
        delete[] this->data;
    }
}



const Pixel Frame::operator[](const Coordinate& index) const {
    // Check if within bounds
    if (index.x >= this->width) {
        throw std::out_of_range("ERROR: const Pixel Frame::operator[](const Coordinate& index) const: X-axis index " + to_string(index.x) + " is out of range for Frame of " + to_string(this->width) + " by " + to_string(this->height) + ".");
    }
    if (index.y >= this->height) {
        throw std::out_of_range("ERROR: const Pixel Frame::operator[](const Coordinate& index) const: Y-axis index " + to_string(index.y) + " is out of range for Frame of " + to_string(this->width) + " by " + to_string(this->height) + ".");
    }

    // Return the correct index Pixel
    double* ptr = this->data + 3 * (index.y * this->width + index.x);
    return Pixel(ptr);
}

Pixel Frame::operator[](const Coordinate& index) {
    // Check if within bounds
    if (index.x >= this->width) {
        throw std::out_of_range("ERROR: Pixel Frame::operator[](const Coordinate& index): X-axis index " + to_string(index.x) + " is out of range for Frame of " + to_string(this->width) + " by " + to_string(this->height) + ".");
    }
    if (index.y >= this->height) {
        throw std::out_of_range("ERROR: Pixel Frame::operator[](const Coordinate& index): Y-axis index " + to_string(index.y) + " is out of range for Frame of " + to_string(this->width) + " by " + to_string(this->height) + ".");
    }

    // Return the correct index Pixel
    double* ptr = this->data + 3 * (index.y * this->width + index.x);
    return Pixel(ptr);
}



void Frame::toPNG(const string& path) const {
    // Generate a 0-255, four channel vector
    vector<unsigned char> raw_image;
    raw_image.resize(this->width * this->height * 4);
    for (unsigned int y = 0; y < this->height; y++) {
        for (unsigned int x = 0; x < this->width; x++) {
            // Store the data as 0-255 Red Green Blue Alhpa
            raw_image[4 * (y * this->width + x) + 0] = (char) (255.0 * this->data[3 * (y * this->width + x)]);
            raw_image[4 * (y * this->width + x) + 1] = (char) (255.0 * this->data[3 * (y * this->width + x) + 1]);
            raw_image[4 * (y * this->width + x) + 2] = (char) (255.0 * this->data[3 * (y * this->width + x) + 2]);
            raw_image[4 * (y * this->width + x) + 3] = 255;
        }
    }

    // Write that to the output file using LodePNG
    unsigned result = lodepng::encode(path.c_str(), raw_image, this->width, this->height);
    if (result != 0) {
        cerr << "Could not write to PNG file: " << lodepng_error_text(result) << endl;
        exit(1);
    }
}



Frame& Frame::operator=(Frame other) {
    // Only do stuff if not ourselves
    if (this != &other) {
        // Check if the size is correct
        if (this->width != other.width || this->height != other.height) {
            throw std::runtime_error("ERROR: Frame& Frame::operator=(Frame other): Cannot copy the value of a Frame with different dimensions.");
        }

        // Swap 'em
        swap(*this, other);
    }

    return *this;
}

Frame& Frame::operator=(Frame&& other) {
    // Only do stuff if not ourselves
    if (this != &other) {
        // Check if the size is correct
        if (this->width != other.width || this->height != other.height) {
            throw std::runtime_error("ERROR: Frame& Frame::operator=(Frame&& other): Cannot copy the value of a Frame with different dimensions.");
        }

        // Swap 'em
        swap(*this, other);
    }

    return *this;
}

void RayTracer::swap(Frame& f1, Frame& f2) {
    using std::swap;

    // Swap the data pointers
    swap(f1.data, f2.data);
}

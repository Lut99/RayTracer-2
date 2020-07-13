/* POINT2.hpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:02:10
 * Last edited:
 *   13/07/2020, 14:35:12
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Point2 class represents a 2-dimensional point not in the world but in
 *   the output frame. To this end, the Point2 class stores positive integers
 *   only rather than decimals who can also be negative.
**/

#ifndef PIXELCOORD_HPP
#define PIXELCOORD_HPP

#include <cstdlib>
#include <ostream>

#include "GPUDev.hpp"

namespace RayTracer {
    /* A class which stores coordinates relevant for an image (integral, non-negative). */
    class Point2 {
    public:
        /* The target x-location. */
        size_t x;
        /* The target y-location. */
        size_t y;

        /* Default constructor for the Point2 class. */
        HOST_DEVICE Point2();
        /* Constructor for the Point2 class which takes the internal coordinates. */
        HOST_DEVICE Point2(size_t x, size_t y);
        /* Copy constructor for the Point2 class. */
        HOST_DEVICE Point2(const Point2& other);
        /* Move constructor for the Point2 class. */
        HOST_DEVICE Point2(Point2&& other);

        #ifdef CUDA
        /* CPU-side default constructor for a GPU-side Point2 object. If ptr == nullptr, does not allocate but only copies to the given memory location. */
        static Point2* GPU_create(void* ptr = nullptr);
        /* CPU-side constructor for a GPU-side Point2 object which takes the internal coordinates. If ptr == nullptr, does not allocate but only copies to the given memory location. */
        static Point2* GPU_create(size_t x, size_t y, void* ptr = nullptr);
        /* CPU-side copy constructor for a GPU-side Point2 object. If ptr == nullptr, does not allocate but only copies to the given memory location. */
        static Point2* GPU_create(const Point2& other, void* ptr = nullptr);
        /* Copies a GPU-side Point2 object back to the CPU as a newly stack-allocated object. */
        static Point2 GPU_copy(Point2* ptr_gpu);
        /* CPU-side destructor for a GPU-side Point2 object. */
        static void GPU_free(Point2* ptr_gpu);
        #endif

        /* Returns true if this Point2 equals another one. */
        HOST_DEVICE inline bool operator==(const Point2& other) const { return this->x == other.x && this->y == other.y; }
        /* Returns true if this Point2 does not equal another one. */
        HOST_DEVICE inline bool operator!=(const Point2& other) const { return this->x != other.x || this->y != other.y; }
        /* Returns true if this Point2 occurs linearly before the given one. In quick terms, only return true if either a) y is smaller or b) the same but x is smaller. */
        HOST_DEVICE inline bool operator<(const Point2& other) const { return this->y < other.y || (this->y == other.y && this->x < other.x); }
        /* Returns true if this Point2 occurs linearly before the given one or equals it. In quick terms, only return true if either a) y is smaller or b) the same but x is smaller or equal. */
        HOST_DEVICE inline bool operator<=(const Point2& other) const { return this->y < other.y || (this->y == other.y && this->x <= other.x); }
        /* Returns true if this Point2 occurs linearly after the given one. In quick terms, only return true if either a) y is larger or b) the same but x is larger. */
        HOST_DEVICE inline bool operator>(const Point2& other) const { return this->y > other.y || (this->y == other.y && this->x > other.x); }
        /* Returns true if this Point2 occurs linearly after the given one or equals it. In quick terms, only return true if either a) y is larger or b) the same but x is larger or equal. */
        HOST_DEVICE inline bool operator>=(const Point2& other) const { return this->y > other.y || (this->y == other.y && this->x >= other.x); }

        /* Allows this Point2 to be added to another to return a new one. */
        HOST_DEVICE inline Point2 operator+(const Point2& other) const { return Point2(this->x + other.x, this->y + other.y); }
        /* Allows this Point2 to be added to another and store the result in this Point2. */
        HOST_DEVICE Point2& operator+=(const Point2& other);

        /* Allows another Point2 to be subtracted from this one to return a new one. */
        HOST_DEVICE inline Point2 operator-(const Point2& other) const { return Point2(this->x - other.x, this->y - other.y); }
        /* Allows another Point2 to be subtracted from this one and store the result in this Point2. */
        HOST_DEVICE Point2& operator-=(const Point2& other);

        /* Allows this Point2 to be multiplied with another to return a new one. */
        HOST_DEVICE inline Point2 operator*(const Point2& other) const { return Point2(this->x * other.x, this->y * other.y); }
        /* Allows this Point2 to be multiplied with another and store the result in this Point2. */
        HOST_DEVICE Point2& operator*=(const Point2& other);

        /* Allows this Point2 to be divided by another to return a new one. Note that this is integer division. */
        HOST_DEVICE inline Point2 operator/(const Point2& other) const { return Point2(this->x / other.x, this->y / other.y); }
        /* Allows this Point2 to be divided by another and store the result in this Point2. Note that this is integer division. */
        HOST_DEVICE Point2& operator/=(const Point2& other);

        /* Allows (non-mutable) access to the internal coordinates by index. */
        HOST_DEVICE size_t operator[](size_t index) const;
        /* Allows (mutable) access to the internal coordinates by index. */
        HOST_DEVICE size_t& operator[](size_t index);

        /* Returns a rebalanced Point2 to a grid of the given width to keep the X within bounds and overflow that to the Y. */
        HOST_DEVICE inline Point2 rebalance(size_t width) const { return Point2(this->x % width, this->y + this->x / width); };

        /* Copy assignment operator for the Point2 class. */
        HOST_DEVICE inline Point2& operator=(const Point2& other) { return *this = Point2(other); }
        /* Move assignment operator for the Point2 class. */
        HOST_DEVICE Point2& operator=(Point2&& other);
        /* Allows the Point2 to be swapped. */
        friend HOST_DEVICE void swap(Point2& p1, Point2& p2);

        /* Allows the Point2 to be printed to a stream. */
        friend inline std::ostream& operator<<(std::ostream& os, const Point2& point);
    };

    /* Allows the Point2 to be swapped. */
    HOST_DEVICE void swap(Point2& p1, Point2& p2);

    /* Allows the Point2 to be printed to a stream. */
    inline std::ostream& operator<<(std::ostream& os, const Point2& point) { return os << "(" << point.x << ", " << point.y << ")"; }
}

#endif

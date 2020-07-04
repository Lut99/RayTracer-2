/* VEC 3.hpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:08:07 PM
 * Last edited:
 *   7/4/2020, 5:59:07 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Vec3-class is where to look for when needing linear algebra
 *   regarding three-dimensional vectors. This particular file focusses on
 *   the CPU-side, but there is an equivalent GPU-side library available as
 *   well.
**/

#ifndef VEC3_HPP
#define VEC3_HPP

#include <cstddef>
#include <stdexcept>
#include <ostream>
#include <cmath>

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    using ::exp;
    using ::sqrt;
    using ::pow;
    using ::fabs;

    class Vec3 {
    private:
        /* Determines if the vector is using external memory or stores it in the object memory. */
        bool is_external;
        /* Local value storage. */
        double local_x, local_y, local_z;
        /* External storage reference. */
        double* external_data;

    public:
        /* The x-coordinate of the vector. */
        double& x;
        /* The y-coordinate of the vector. */
        double& y;
        /* The z-coordinate of the vector. */
        double& z;

        /* Default constructor for the Vec3-class which initializes a vector with all-zero elements. */
        HOST_DEVICE Vec3();
        /* Constructor which takes three elements for the vector to be initialized with. */
        HOST_DEVICE Vec3(double x, double y, double z);
        #ifdef CUDA
        /* GPU-side constructor for using external memory. */
        __device__ Vec3(void* data);
        #endif
        /* Copy constructor for the Vec3-class. */
        HOST_DEVICE Vec3(const Vec3& other);
        /* Move constructor for the Vec3-class. */
        HOST_DEVICE Vec3(Vec3&& other);

        /* Compares if two vectors are equal. */
        HOST_DEVICE inline bool operator==(const Vec3& other) const { return this->x == other.x && this->y == other.y && this->z == other.z; }
        /* Compares if two vectors are not equal. */
        HOST_DEVICE inline bool operator!=(const Vec3& other) const { return this->x != other.x || this->y != other.y || this->z != other.z; }
        /* Compares if all elements in a vector are less than a given constant. */
        HOST_DEVICE inline bool operator<(const double c) const { return this->x < c && this->y < c && this->z < c; }
        /* Compares if all elements in a vector are less than or equal to a given constant. */
        HOST_DEVICE inline bool operator<=(const double c) const { return this->x <= c && this->y <= c && this->z <= c; }
        /* Compares if all elements in a vector are greater than a given constant. */
        HOST_DEVICE inline bool operator>(const double c) const { return this->x > c && this->y > c && this->z > c; }
        /* Compares if all elements in a vector are greater than or equal to a given constant. */
        HOST_DEVICE inline bool operator>=(const double c) const { return this->x >= c && this->y >= c && this->z >= c; }

        /* Adds a constant to all elements in the vector and returns the result as a new one. */
        HOST_DEVICE inline Vec3 operator+(const double c) const { return Vec3(this->x + c, this->y + c, this->z + c); }
        /* Adds a constant to all elements in the vector and returns the result in this one. */
        HOST_DEVICE Vec3& operator+=(const double c);
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in a new one. */
        HOST_DEVICE inline Vec3 operator+(const Vec3& other) const { return Vec3(this->x + other.x, this->y + other.y, this->z + other.z); }
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in this one. */
        HOST_DEVICE Vec3& operator+=(const Vec3& other);

        /* Return a copy of this vector with all elements negated. */
        HOST_DEVICE inline Vec3 operator-() const { return Vec3(-this->x, -this->y, -this->z); }
        /* Subtracts a constant from all elements in the vector and returns the result as a new one. */
        HOST_DEVICE inline Vec3 operator-(const double c) const { return Vec3(this->x - c, this->y - c, this->z - c); }
        /* Subtracts a constant from all elements in the vector and returns the result in this one. */
        HOST_DEVICE Vec3& operator-=(const double c);
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in a new one. */
        HOST_DEVICE inline Vec3 operator-(const Vec3& other) const { return Vec3(this->x - other.x, this->y - other.y, this->z - other.z); }
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in this one. */
        HOST_DEVICE Vec3& operator-=(const Vec3& other);

        /* Multiplies a constant with all elements in the vector and returns the result as a new one. */
        HOST_DEVICE inline Vec3 operator*(const double c) const { return Vec3(this->x * c, this->y * c, this->z * c); }
        /* Multiplies a constant with all elements in the vector and returns the result in this one. */
        HOST_DEVICE Vec3& operator*=(const double c);
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in a new one. */
        HOST_DEVICE inline Vec3 operator*(const Vec3& other) const { return Vec3(this->x * other.x, this->y * other.y, this->z * other.z); }
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in this one. */
        HOST_DEVICE Vec3& operator*=(const Vec3& other);

        /* Inverts all elements in this vector (1/x) and returns a copy. */
        HOST_DEVICE inline Vec3 inv() const { return Vec3(1 / this->x, 1 / this->y, 1 / this->z); }
        /* Divides all elements in this vector by a constant and returns the result as a new one. */
        HOST_DEVICE inline Vec3 operator/(const double c) const { return Vec3(this->x / c, this->y / c, this->z / c); }
        /* Divides all elements in this vector by a constant and returns the result in this one. */
        HOST_DEVICE inline Vec3& operator/=(const double c) { return *this *= (1 / c); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in a new one. */
        HOST_DEVICE inline Vec3 operator/(const Vec3& other) const { return Vec3(this->x / other.x, this->y / other.y, this->z / other.z); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in this one. */
        HOST_DEVICE inline Vec3& operator/=(const Vec3& other) { return *this *= other.inv(); }

        /* Returns the sum of all elements of this vector. */
        HOST_DEVICE inline double sum() const { return this->x + this->y + this->z; }
        /* Returns the length of this vector. */
        HOST_DEVICE inline double length() const { return sqrt(this->x * this->x + this->y * this->y + this->z * this->z); }
        /* Returns the squared length of this vector. */
        HOST_DEVICE inline double length_pow2() const { return this->x * this->x + this->y * this->y + this->z * this->z; }
        /* Returns the normalized version of this vector. */
        HOST_DEVICE inline Vec3 normalize() const { return Vec3(*this) / this->length(); }

        /* Copy assignment operator for the Vec3-class. */
        HOST_DEVICE Vec3& operator=(Vec3 other);
        /* Move assignment operator for the Vec3-class. */
        HOST_DEVICE Vec3& operator=(Vec3&& other);
        /* Swap operator for the Vec3-class. */
        friend HOST_DEVICE void swap(Vec3& v1, Vec3& v2);

        /* Returns x, y or z based on their index (non-mutable). */
        HOST_DEVICE double operator[](const size_t i) const;
        /* Returns x, y or z based on their index (mutable). */
        HOST_DEVICE double& operator[](const size_t i);

        #ifdef CUDA
        /* Copies the Vec3 object to the GPU. */
        void* toGPU() const;
        /* Copies the Vec3 object from the GPU to a new CPU-side object. */
        static Vec3 fromGPU(void* ptr_gpu);
        #endif

        /* Allows the vector to be printed to a stream. */
        friend std::ostream& operator<<(std::ostream& os, const Vec3& vec);
    };

    /* Swaps two Vec3-objects. */
    HOST_DEVICE void swap(Vec3& v1, Vec3& v2);

    /* Allows the vector to be printed to a stream (CPU-only). */
    std::ostream& operator<<(std::ostream& os, const Vec3& vec);

    /* Performs the sum-operation to sum all elements of a vector. */
    HOST_DEVICE inline double sum(const Vec3& vec) { return vec.x + vec.y + vec.z; }
    /* Performs the exp-operation on each element of the vector. */
    HOST_DEVICE inline Vec3 exp(const Vec3& vec) { return Vec3(exp(vec.x), exp(vec.y), exp(vec.z)); }
    /* Performs the sqrt-operation on each element of the vector. */
    HOST_DEVICE inline Vec3 sqrt(const Vec3& vec) { return Vec3(sqrt(vec.x), sqrt(vec.y), sqrt(vec.z)); }
    /* Performs the pow-operation on each element of the vector. */
    HOST_DEVICE inline Vec3 pow(const Vec3& vec, const double c) { return Vec3(pow(vec.x, c), pow(vec.y, c), pow(vec.z, c)); }
    /* Performs the fabs-operation on each element of the vector. */
    HOST_DEVICE inline Vec3 fabs(const Vec3& vec) { return Vec3(fabs(vec.x), fabs(vec.y), fabs(vec.z)); }
    /* Performs a dot product between two vectors. */
    HOST_DEVICE inline double dot(const Vec3& v1, const Vec3& v2) { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
    /* Performs the cross product between two vectors. */
    HOST_DEVICE inline Vec3 cross(const Vec3& v1, const Vec3& v2) { return Vec3(v1.y * v2.z - v1.z * v2.y,
                                                                                v1.z * v2.x - v1.x * v2.z,
                                                                                v1.x * v2.y - v1.y * v2.x); }
}

#endif

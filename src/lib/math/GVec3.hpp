/* GVEC 3.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 2:18:03 PM
 * Last edited:
 *   7/1/2020, 4:41:12 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file provides the same math as Vec3 does, except that all its
 *   functions live on the GPU. Additionally, it also provides the means to
 *   copy a given Vec3 to the GPU as GVec3 and to copy GVec3 back as Vec3.
**/

#ifndef GVEC3_HPP
#define GVEC3_HPP

#include <cstddef>
#include <cmath>

#include "Vec3.hpp"

namespace RayTracer {
    using ::exp;
    using ::sqrt;
    using ::pow;
    using ::fabs;

    class GVec3 {
    private:
        /* Indicates if x, y and z point to these. */
        bool is_local;
        /* Storage for when none is assigned. */
        double dx, dy, dz;
    public:
        /* Pointer to the container of the data. */
        double* data;
        /* The x-coordinate of the vector. */
        double& x;
        /* The y-coordinate of the vector. */
        double& y;
        /* The z-coordinate of the vector. */
        double& z;

        /* Default constructor for the GVec3-class which initializes a vector with all-zero elements. */
        __device__ GVec3();
        /* Constructor which takes three elements for the vector to be initialized with. */
        __device__ GVec3(double x, double y, double z);
        /* Constructor which is based on a memory address. Uses this memory instead of the normal x, y, z. */
        __device__ GVec3(void* data);
        /* Copy constructor for the GVec3 class. */
        __device__ GVec3(const GVec3& other);
        /* Move constructor for the GVec3 class. */
        __device__ GVec3(GVec3&& other);

        /* Compares if two vectors are equal. */
        __device__ bool operator==(const GVec3& other) const { return this->x == other.x && this->y == other.y && this->z == other.z; }
        /* Compares if two vectors are not equal. */
        __device__ bool operator!=(const GVec3& other) const { return this->x != other.x || this->y != other.y || this->z != other.z; }
        /* Compares if all elements in a vector are less than a given constant. */
        __device__ bool operator<(const double c) const { return this->x < c && this->y < c && this->z < c; }
        /* Compares if all elements in a vector are less than or equal to a given constant. */
        __device__ bool operator<=(const double c) const { return this->x <= c && this->y <= c && this->z <= c; }
        /* Compares if all elements in a vector are greater than a given constant. */
        __device__ bool operator>(const double c) const { return this->x > c && this->y > c && this->z > c; }
        /* Compares if all elements in a vector are greater than or equal to a given constant. */
        __device__ bool operator>=(const double c) const { return this->x >= c && this->y >= c && this->z >= c; }
        

        /* Adds a constant to all elements in the vector and returns the result as a new one. */
        __device__ GVec3 operator+(const double c) const { return GVec3(this->x + c, this->y + c, this->z + c); }
        /* Adds a constant to all elements in the vector and returns the result in this one. */
        __device__ GVec3& operator+=(const double c);
        /* Adds another GVec3-object to this vector (element-wise) and returns the result in a new one. */
        __device__ GVec3 operator+(const GVec3& other) const { return GVec3(this->x + other.x, this->y + other.y, this->z + other.z); }
        /* Adds another GVec3-object to this vector (element-wise) and returns the result in this one. */
        __device__ GVec3& operator+=(const GVec3& other);

        /* Return a copy of this vector with all elements negated. */
        __device__ GVec3 operator-() const { return GVec3(-this->x, -this->y, -this->z); }
        /* Subtracts a constant from all elements in the vector and returns the result as a new one. */
        __device__ GVec3 operator-(const double c) const { return GVec3(this->x - c, this->y - c, this->z - c); }
        /* Subtracts a constant from all elements in the vector and returns the result in this one. */
        __device__ GVec3& operator-=(const double c);
        /* Subtracts another GVec3-object from this vector (element-wise) and returns the result in a new one. */
        __device__ GVec3 operator-(const GVec3& other) const { return GVec3(this->x - other.x, this->y - other.y, this->z - other.z); }
        /* Subtracts another GVec3-object from this vector (element-wise) and returns the result in this one. */
        __device__ GVec3& operator-=(const GVec3& other);

        /* Multiplies a constant with all elements in the vector and returns the result as a new one. */
        __device__ GVec3 operator*(const double c) const { return GVec3(this->x * c, this->y * c, this->z * c); }
        /* Multiplies a constant with all elements in the vector and returns the result in this one. */
        __device__ GVec3& operator*=(const double c);
        /* Multiplies another GVec3-object with this vector (element-wise) and returns the result in a new one. */
        __device__ GVec3 operator*(const GVec3& other) const { return GVec3(this->x * other.x, this->y * other.y, this->z * other.z); }
        /* Multiplies another GVec3-object with this vector (element-wise) and returns the result in this one. */
        __device__ GVec3& operator*=(const GVec3& other);

        /* Inverts all elements in this vector (1/x) and returns a copy. */
        __device__ GVec3 inv() const { return GVec3(1 / this->x, 1 / this->y, 1 / this->z); }
        /* Divides all elements in this vector by a constant and returns the result as a new one. */
        __device__ GVec3 operator/(const double c) const { return GVec3(this->x / c, this->y / c, this->z / c); }
        /* Divides all elements in this vector by a constant and returns the result in this one. */
        __device__ GVec3& operator/=(const double c) { return *this *= (1 / c); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in a new one. */
        __device__ GVec3 operator/(const GVec3& other) const { return GVec3(this->x / other.x, this->y / other.y, this->z / other.z); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in this one. */
        __device__ GVec3& operator/=(const GVec3& other) { return *this *= other.inv(); }

        /* Returns the sum of all elements of this vector. */
        __device__ double sum() const { return this->x + this->y + this->z; }
        /* Returns the length of this vector. */
        __device__ double length() const { return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z); }
        /* Returns the squared length of this vector. */
        __device__ double length_pow2() const { return this->x * this->x + this->y * this->y + this->z * this->z; }

        /* Returns x, y or z based on their index (non-mutable). */
        __device__ double operator[](const size_t i) const;
        /* Returns x, y or z based on their index (mutable). */
        __device__ double& operator[](const size_t i);

        /* Copy assign operator for the GVec3 class. */
        __device__ GVec3& operator=(GVec3 other);
        /* Move assign operator for the GVec3 class. */
        __device__ GVec3& operator=(GVec3&& other);
        /* Swap operator for the GVec3 class. */
        friend __device__ void swap(GVec3& vec1, GVec3& vec2);

        /* Copies given CPU-side vector to the GPU. */
        __host__ static void* toGPU(const Vec3& vec);
        /* Copies given GPU-side pointer of Vec3-memory back to the CPU and creates a new Vec3-object with it. Note that it also frees the GPU memory. */
        __host__ static Vec3 fromGPU(void* ptr);

        /* Allows exp to work on the vector. */
        friend __device__ GVec3 exp(const GVec3& vec);
        /* Allows sqrt to work on the vector. */
        friend __device__ GVec3 sqrt(const GVec3& vec);
        /* Allows pow to work on the vector. */
        friend __device__ GVec3 pow(const GVec3& vec);
        /* Allows fabs to work on the vector. */
        friend __device__ GVec3 fabs(const GVec3& vec);

        /* Writes the vector to stdout */
        __device__ void print() const;
    };

    /* Swap operator for the GVec3 class. */
    __device__ void swap(GVec3& vec1, GVec3& vec2);

    /* Performs the exp-operation on each element of the vector. */
    __device__ GVec3 exp(const GVec3& vec);
    /* Performs the sqrt-operation on each element of the vector. */
    __device__ GVec3 sqrt(const GVec3& vec);
    /* Performs the pow-operation on each element of the vector. */
    __device__ GVec3 pow(const GVec3& vec, const double c);
    /* Performs the fabs-operation on each element of the vector. */
    __device__ GVec3 fabs(const GVec3& vec);
}

#endif

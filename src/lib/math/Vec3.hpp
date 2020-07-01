/* VEC 3.hpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:08:07 PM
 * Last edited:
 *   7/1/2020, 4:35:47 PM
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

namespace RayTracer {
    using ::exp;
    using ::sqrt;
    using ::pow;
    using ::fabs;

    class Vec3 {
    public:
        /* The coordinates of the vector. */
        double x, y, z;

        /* Default constructor for the Vec3-class which initializes a vector with all-zero elements. */
        Vec3();
        /* Constructor which takes three elements for the vector to be initialized with. */
        Vec3(double x, double y, double z);

        /* Compares if two vectors are equal. */
        inline bool operator==(const Vec3& other) const { return this->x == other.x && this->y == other.y && this->z == other.z; }
        /* Compares if two vectors are not equal. */
        inline bool operator!=(const Vec3& other) const { return this->x != other.x || this->y != other.y || this->z != other.z; }
        /* Compares if all elements in a vector are less than a given constant. */
        inline bool operator<(const double c) const { return this->x < c && this->y < c && this->z < c; }
        /* Compares if all elements in a vector are less than or equal to a given constant. */
        inline bool operator<=(const double c) const { return this->x <= c && this->y <= c && this->z <= c; }
        /* Compares if all elements in a vector are greater than a given constant. */
        inline bool operator>(const double c) const { return this->x > c && this->y > c && this->z > c; }
        /* Compares if all elements in a vector are greater than or equal to a given constant. */
        inline bool operator>=(const double c) const { return this->x >= c && this->y >= c && this->z >= c; }
        

        /* Adds a constant to all elements in the vector and returns the result as a new one. */
        inline Vec3 operator+(const double c) const { return Vec3(this->x + c, this->y + c, this->z + c); }
        /* Adds a constant to all elements in the vector and returns the result in this one. */
        Vec3& operator+=(const double c);
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator+(const Vec3& other) const { return Vec3(this->x + other.x, this->y + other.y, this->z + other.z); }
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in this one. */
        Vec3& operator+=(const Vec3& other);

        /* Return a copy of this vector with all elements negated. */
        inline Vec3 operator-() const { return Vec3(-this->x, -this->y, -this->z); }
        /* Subtracts a constant from all elements in the vector and returns the result as a new one. */
        inline Vec3 operator-(const double c) const { return Vec3(this->x - c, this->y - c, this->z - c); }
        /* Subtracts a constant from all elements in the vector and returns the result in this one. */
        Vec3& operator-=(const double c);
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator-(const Vec3& other) const { return Vec3(this->x - other.x, this->y - other.y, this->z - other.z); }
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in this one. */
        Vec3& operator-=(const Vec3& other);

        /* Multiplies a constant with all elements in the vector and returns the result as a new one. */
        inline Vec3 operator*(const double c) const { return Vec3(this->x * c, this->y * c, this->z * c); }
        /* Multiplies a constant with all elements in the vector and returns the result in this one. */
        Vec3& operator*=(const double c);
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator*(const Vec3& other) const { return Vec3(this->x * other.x, this->y * other.y, this->z * other.z); }
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in this one. */
        Vec3& operator*=(const Vec3& other);

        /* Inverts all elements in this vector (1/x) and returns a copy. */
        inline Vec3 inv() const { return Vec3(1 / this->x, 1 / this->y, 1 / this->z); }
        /* Divides all elements in this vector by a constant and returns the result as a new one. */
        inline Vec3 operator/(const double c) const { return Vec3(this->x / c, this->y / c, this->z / c); }
        /* Divides all elements in this vector by a constant and returns the result in this one. */
        inline Vec3& operator/=(const double c) { return *this *= (1 / c); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator/(const Vec3& other) const { return Vec3(this->x / other.x, this->y / other.y, this->z / other.z); }
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in this one. */
        inline Vec3& operator/=(const Vec3& other) { return *this *= other.inv(); }

        /* Returns the sum of all elements of this vector. */
        inline double sum() const { return this->x + this->y + this->z; }
        /* Returns the length of this vector. */
        inline double length() const { return sqrt(this->x * this->x + this->y * this->y + this->z * this->z); }
        /* Returns the squared length of this vector. */
        inline double length_pow2() const { return this->x * this->x + this->y * this->y + this->z * this->z; }

        /* Returns x, y or z based on their index (non-mutable). */
        double operator[](const size_t i) const;
        /* Returns x, y or z based on their index (mutable). */
        double& operator[](const size_t i);

        /* Allows exp to work on the vector. */
        friend Vec3 exp(const Vec3& vec);
        /* Allows sqrt to work on the vector. */
        friend Vec3 sqrt(const Vec3& vec);
        /* Allows pow to work on the vector. */
        friend Vec3 pow(const Vec3& vec);
        /* Allows fabs to work on the vector. */
        friend Vec3 fabs(const Vec3& vec);

        /* Allows the vector to be printed to a stream. */
        friend std::ostream& operator<<(std::ostream& os, const Vec3& vec);
    };

    /* Performs the exp-operation on each element of the vector. */
    Vec3 exp(const Vec3& vec);
    /* Performs the sqrt-operation on each element of the vector. */
    Vec3 sqrt(const Vec3& vec);
    /* Performs the pow-operation on each element of the vector. */
    Vec3 pow(const Vec3& vec, const double c);
    /* Performs the fabs-operation on each element of the vector. */
    Vec3 fabs(const Vec3& vec);

    /* Allows the vector to be printed to a stream. */
    std::ostream& operator<<(std::ostream& os, const Vec3& vec);
}

#endif

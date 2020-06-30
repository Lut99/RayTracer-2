/* VEC 3.hpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:08:07 PM
 * Last edited:
 *   6/30/2020, 6:06:51 PM
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

namespace RayTracer {
    class Vec3 {
    public:
        /* The coordinates of the vector. */
        float x, y, z;

        /* Default constructor for the Vec3-class which initializes a vector with all-zero elements. */
        Vec3();
        /* Constructor which takes three elements for the vector to be initialized with. */
        Vec3(float x, float y, float z);

        /* Compares if two vectors are equal. */
        inline bool operator==(const Vec3& other) const;
        /* Compares if two vectors are not equal. */
        inline bool operator!=(const Vec3& other) const;
        /* Compares if all elements in a vector are less than a given constant. */
        inline bool operator<(const float c) const;
        /* Compares if all elements in a vector are less than or equal to a given constant. */
        inline bool operator<=(const float c) const;
        /* Compares if all elements in a vector are greater than a given constant. */
        inline bool operator>(const float c) const;
        /* Compares if all elements in a vector are greater than or equal to a given constant. */
        inline bool operator>=(const float c) const;
        

        /* Adds a constant to all elements in the vector and returns the result as a new one. */
        inline Vec3 operator+(const float c) const;
        /* Adds a constant to all elements in the vector and returns the result in this one. */
        Vec3& operator+=(const float c);
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator+(const Vec3& other) const;
        /* Adds another Vec3-object to this vector (element-wise) and returns the result in this one. */
        Vec3& operator+=(const Vec3& other);

        /* Return a copy of this vector with all elements negated. */
        inline Vec3 operator-() const;
        /* Subtracts a constant from all elements in the vector and returns the result as a new one. */
        inline Vec3 operator-(const float c) const;
        /* Subtracts a constant from all elements in the vector and returns the result in this one. */
        Vec3& operator-=(const float c);
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator-(const Vec3& other) const;
        /* Subtracts another Vec3-object from this vector (element-wise) and returns the result in this one. */
        Vec3& operator-=(const Vec3& other);

        /* Multiplies a constant with all elements in the vector and returns the result as a new one. */
        inline Vec3 operator*(const float c) const;
        /* Multiplies a constant with all elements in the vector and returns the result in this one. */
        Vec3& operator*=(const float c);
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator*(const Vec3& other) const;
        /* Multiplies another Vec3-object with this vector (element-wise) and returns the result in this one. */
        Vec3& operator*=(const Vec3& other);

        /* Inverts all elements in this vector (1/x) and returns a copy. */
        inline Vec3 inv() const;
        /* Divides all elements in this vector by a constant and returns the result as a new one. */
        inline Vec3 operator/(const float c) const;
        /* Divides all elements in this vector by a constant and returns the result in this one. */
        inline Vec3& operator/=(const float c);
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in a new one. */
        inline Vec3 operator/(const Vec3& other) const;
        /* Divides all elements in this vector by another vector (element-wise) and returns the result in this one. */
        inline Vec3& operator/=(const Vec3& other);

        /* Takes the square root of each element and returns a copy. */
        inline Vec3 sqrt() const;
        /* Squares all elements in this vector and returns a copy. */
        inline Vec3 pow2() const;
        /* Does every element to the power of c (x^c) and returns a copy. */
        inline Vec3 pow(const float c) const;

        /* Returns the sum of all elements of this vector. */
        inline float sum() const;
        /* Returns the length of this vector. */
        inline float length() const;
        /* Returns the squared length of this vector. */
        inline float length_pow2() const;

        /* Returns x, y or z based on their index (non-mutable). */
        float operator[](const size_t i) const;
        /* Returns x, y or z based on their index (mutable). */
        float& operator[](const size_t i);

        /* Allows the vector to be printed to a stream. */
        friend inline ostream& operator<<(ostream& os, const Vec3& vec);
    };

    /* Allows the vector to be printed to a stream. */
    inline ostream& operator<<(ostream& os, const Vec3& vec);
}

#endif

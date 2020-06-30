/* VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:09:06 PM
 * Last edited:
 *   6/30/2020, 6:13:49 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Vec3-class is where to look for when needing linear algebra
 *   regarding three-dimensional vectors. This particular file focusses on
 *   the CPU-side, but there is an equivalent GPU-side library available as
 *   well
**/

#include <cmath>

#include "Vec3.hpp"

using namespace std;
using namespace RayTracer;


Vec3::Vec3() :
    x(0),
    y(0),
    z(0)
{}

Vec3::Vec3(float x, float y, float z) :
    x(x),
    y(y),
    z(z)
{}



inline bool Vec3::operator==(const Vec3& other) const {
    return this->x == other.x && this->y == other.y && this->z == other.z;
}

inline bool Vec3::operator!=(const Vec3& other) const {
    return this->x != other.x || this->y != other.y || this->z != other.z;
}

inline bool Vec3::operator<(const float c) const {
    return this->x < c && this->y < c && this->z < c;
}

inline bool Vec3::operator<=(const float c) const {
    return this->x <= c && this->y <= c && this->z <= c;
}

inline bool Vec3::operator>(const float c) const {
    return this->x > c && this->y > c && this->z > c;
}

inline bool Vec3::operator>=(const float c) const {
    return this->x >= c && this->y >= c && this->z >= c;
}



inline Vec3 Vec3::operator+(const float c) const {
    return Vec3(this->x + c, this->y + c, this->z + c);
}

Vec3& Vec3::operator+=(const float c) {
    this->x += c;
    this->y += c;
    this->z += c;
    return *this;
}

inline Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(this->x - other.x, this->y - other.y, this->z - other.z);
}

Vec3& Vec3::operator-=(const Vec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



inline Vec3 Vec3::operator-() const {
    return Vec3(-this->x, -this->y, -this->z);
}

inline Vec3 Vec3::operator-(const float c) const {
    return Vec3(this->x - c, this->y - c, this->z - c);
}

Vec3& Vec3::operator-=(const float c) {
    this->x -= c;
    this->y -= c;
    this->z -= c;
    return *this;
}

inline Vec3 Vec3::operator-(const Vec3& other) const {
    return Vec3(this->x - other.x, this->y - other.y, this->z - other.z);
}

Vec3& Vec3::operator-=(const Vec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



inline Vec3 Vec3::operator*(const float c) const {
    return Vec3(this->x * c, this->y * c, this->z * c);
}

Vec3& Vec3::operator*=(const float c) {
    this->x *= c;
    this->y *= c;
    this->z *= c;
    return *this;
}

inline Vec3 Vec3::operator*(const Vec3& other) const {
    return Vec3(this->x * other.x, this->y * other.y, this->z * other.z);
}

Vec3& Vec3::operator*=(const Vec3& other) {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
}



inline Vec3 Vec3::inv() const {
    return Vec3(1 / this->x, 1 / this->y, 1 / this->z);
}

inline Vec3 Vec3::operator/(const float c) const {
    return Vec3(this->x / c, this->y / c, this->z / c);
}

inline Vec3& Vec3::operator/=(const float c) {
    return *this *= (1 / c);
}

inline Vec3 Vec3::operator/(const Vec3& other) const {
    return Vec3(this->x / other.x, this->y / other.y, this->z / other.z);
}

inline Vec3& Vec3::operator/=(const Vec3& other) {
    return *this *= other.inv();
}



inline Vec3 Vec3::sqrt() const {
    return Vec3(sqrtf(this->x), sqrtf(this->y), sqrtf(this->z));
}

inline Vec3 Vec3::pow2() const {
    return Vec3(this->x * this->x, this->y * this->y, this->z * this->z);
}

inline Vec3 Vec3::pow(const float c) const {
    return Vec3(powf(this->x, c), powf(this->y, c), powf(this->z, c));
}



inline float Vec3::sum() const {
    return this->x + this->y + this->z;
}

inline float Vec3::length() const {
    return sqrtf(this->x * this->x + this->y * this->y + this->z * this->z);
}

inline float Vec3::length_pow2() const {
    return this->x * this->x + this->y * this->y + this->z * this->z;
}



float Vec3::operator[](const size_t i) const {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    throw out_of_range("ERROR: float Vec3::operator[](const size-t i) const: Index " + to_string(i) + " is out of bounds for vector of length 3.");
}

float& Vec3::operator[](const size_t i) {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    throw out_of_range("ERROR: float Vec3::operator[](const size-t i) const: Index " + to_string(i) + " is out of bounds for vector of length 3.");
}



inline ostream& RayTracer::operator<<(ostream& os, const Vec3& vec) {
    return os << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
}

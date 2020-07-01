/* VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:09:06 PM
 * Last edited:
 *   7/1/2020, 12:11:37 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Vec3-class is where to look for when needing linear algebra
 *   regarding three-dimensional vectors. This particular file focusses on
 *   the CPU-side, but there is an equivalent GPU-side library available as
 *   well
**/

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



Vec3& Vec3::operator+=(const float c) {
    this->x += c;
    this->y += c;
    this->z += c;
    return *this;
}

Vec3& Vec3::operator+=(const Vec3& other) {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
}



Vec3& Vec3::operator-=(const float c) {
    this->x -= c;
    this->y -= c;
    this->z -= c;
    return *this;
}

Vec3& Vec3::operator-=(const Vec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



Vec3& Vec3::operator*=(const float c) {
    this->x *= c;
    this->y *= c;
    this->z *= c;
    return *this;
}

Vec3& Vec3::operator*=(const Vec3& other) {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
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

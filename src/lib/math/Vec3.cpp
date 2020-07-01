/* VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:09:06 PM
 * Last edited:
 *   7/1/2020, 4:15:30 PM
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

Vec3::Vec3(double x, double y, double z) :
    x(x),
    y(y),
    z(z)
{}



Vec3& Vec3::operator+=(const double c) {
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



Vec3& Vec3::operator-=(const double c) {
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



Vec3& Vec3::operator*=(const double c) {
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



double Vec3::operator[](const size_t i) const {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    throw out_of_range("ERROR: double Vec3::operator[](const size_t i) const: Index " + to_string(i) + " is out of bounds for vector of length 3.");
}

double& Vec3::operator[](const size_t i) {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    throw out_of_range("ERROR: double& Vec3::operator[](const size_t i): Index " + to_string(i) + " is out of bounds for vector of length 3.");
}



/* Performs the exp-operation on each element of the vector. */
Vec3 RayTracer::exp(const Vec3& vec) {
    return Vec3(exp(vec.x), exp(vec.y), exp(vec.z));
}
/* Performs the sqrt-operation on each element of the vector. */
Vec3 RayTracer::sqrt(const Vec3& vec) {
    return Vec3(sqrt(vec.x), sqrt(vec.y), sqrt(vec.z));
}
/* Performs the pow-operation on each element of the vector. */
Vec3 RayTracer::pow(const Vec3& vec, const double c) {
    return Vec3(pow(vec.x, c), pow(vec.y, c), pow(vec.z, c));
}
/* Performs the fabs-operation on each element of the vector. */
Vec3 RayTracer::fabs(const Vec3& vec) {
    return Vec3(fabs(vec.x), fabs(vec.y), fabs(vec.z));
}



/* Allows the vector to be printed to a stream. */
std::ostream& RayTracer::operator<<(std::ostream& os, const Vec3& vec) {
    return os << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
}

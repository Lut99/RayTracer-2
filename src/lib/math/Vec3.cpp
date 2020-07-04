/* VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:09:06 PM
 * Last edited:
 *   7/4/2020, 6:06:18 PM
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


HOST_DEVICE Vec3::Vec3() :
    x(local_x),
    y(local_y),
    z(local_z)
{
    this->is_external = false;
    this->local_x = 0;
    this->local_y = 0;
    this->local_z = 0;
}

HOST_DEVICE Vec3::Vec3(double x, double y, double z) :
    x(local_x),
    y(local_y),
    z(local_z)
{
    this->is_external = false;
    this->local_x = x;
    this->local_y = y;
    this->local_z = z;
}

#ifdef CUDA
__device__ Vec3::Vec3(void* data) :
    external_data((double*) data),
    x(this->external_data[0]),
    y(this->external_data[1]),
    z(this->external_data[2])
{
    this->is_external = true;
}
#endif

HOST_DEVICE Vec3::Vec3(const Vec3& other) :
    x(local_x),
    y(local_y),
    z(local_z)
{
    this->is_external = false;
    this->local_x = other.x;
    this->local_y = other.y;
    this->local_z = other.z;
}

HOST_DEVICE Vec3::Vec3(Vec3&& other) :
    x(local_x),
    y(local_y),
    z(local_z)
{
    this->is_external = false;
    this->local_x = other.x;
    this->local_y = other.y;
    this->local_z = other.z;
}



HOST_DEVICE Vec3& Vec3::operator+=(const double c) {
    this->x += c;
    this->y += c;
    this->z += c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator+=(const Vec3& other) {
    this->x += other.x;
    this->y += other.y;
    this->z += other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator-=(const double c) {
    this->x -= c;
    this->y -= c;
    this->z -= c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator-=(const Vec3& other) {
    this->x -= other.x;
    this->y -= other.y;
    this->z -= other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator*=(const double c) {
    this->x *= c;
    this->y *= c;
    this->z *= c;
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator*=(const Vec3& other) {
    this->x *= other.x;
    this->y *= other.y;
    this->z *= other.z;
    return *this;
}



HOST_DEVICE Vec3& Vec3::operator=(Vec3 other) {
    // Never ourselves, just swap lmao
    swap(*this, other);
    return *this;
}

HOST_DEVICE Vec3& Vec3::operator=(Vec3&& other) {
    // Only swap if ourselvse
    if (this == &other) {
        swap(*this, other);
    }
    return *this;
}

HOST_DEVICE void RayTracer::swap(Vec3& v1, Vec3& v2) {
    // Simply swap x, y and z; the referenced items will follow, and we needn't change whether it's external or not.
    double t = v1.x;
    v1.x = v2.x;
    v2.x = t;

    t = v1.y;
    v1.y = v2.y;
    v2.y = t;

    t = v1.z;
    v1.z = v2.z;
    v2.z = t;
}



HOST_DEVICE double Vec3::operator[](const size_t i) const {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double Vec3::operator[](const size_t i) const: Index %lu is out of bounds for vector of length 3.", i);
    return -INFINITY;
}

HOST_DEVICE double& Vec3::operator[](const size_t i) {
    if (i == 0) { return this->x; }
    else if (i == 1) { return this->y; }
    else if (i == 2) { return this->z; }
    
    // Else...
    printf("ERROR: double& Vec3::operator[](const size_t i): Index %lu is out of bounds for vector of length 3.", i);
    return this->x;
}



/* Allows the vector to be printed to a stream. */
std::ostream& RayTracer::operator<<(std::ostream& os, const Vec3& vec) {
    return os << "[" << vec.x << ", " << vec.y << ", " << vec.z << "]";
}

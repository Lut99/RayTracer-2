/* RAY ITERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 15:53:13
 * Last edited:
 *   09/07/2020, 16:54:20
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayIterator class is used as a wrapper around a Camera to easily
 *   iterate over Rays. Note that his will likely only be used on the CPU,
 *   and so contains no functionality to exist on the GPU whatsoever.
**/

#include "RayIterator.hpp"

using namespace std;
using namespace RayTracer;


/***** CONST ITERATOR *****/
RayIterator::const_iterator::const_iterator(const RayIterator* data) :
    data(data),
    pos(0)
{}

RayIterator::const_iterator::const_iterator(const RayIterator* data, size_t pos) :
    data(data),
    pos(pos)
{}



RayIterator::const_iterator& RayIterator::const_iterator::operator++() {
    // Increment this pos, return
    this->pos++;
    return *this;
}

RayIterator::const_iterator& RayIterator::const_iterator::operator+=(size_t n) {
    // Increment this pos, return
    this->pos += n;
    return *this;
}





/***** RAY ITERATOR *****/
RayIterator::RayIterator(Camera& camera, size_t n_rays) :
    camera(camera),
    n_rays(n_rays)
{}

RayIterator::RayIterator(const RayIterator& other) :
    camera(other.camera),
    n_rays(other.n_rays)
{}

RayIterator::RayIterator(RayIterator&& other) :
    camera(other.camera),
    n_rays(other.n_rays)
{}



RayIterator& RayIterator::operator=(RayIterator&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

void RayTracer::swap(RayIterator& ri1, RayIterator& ri2) {
    using std::swap;

    // Swap the camera
    swap(ri1.camera, ri2.camera);
    // Swap the number of rays
    swap(ri1.n_rays, ri2.n_rays);
}

/* RAY GENERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:46:10
 * Last edited:
 *   08/07/2020, 21:55:38
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayGenerator-class is used to create the Rays which will be iterated
 *   over. It spawns RayBatch-objects, which are collections of Rays of a set
 *   number, which will mediate between being able to do multiple rays at the
 *   same time and keeping memory usage within bounds.
 * 
 *   Note that the RayGenerator itself will not live on the GPU, so only the
 *   RayBatch will have GPU-constructors and deconstructors.
**/

#include "RayGenerator.hpp"

using namespace std;
using namespace RayTracer;


/***** RAYBATCH ITERATOR *****/
RayBatch::iterator::iterator(RayBatch* batch) :
    data(batch),
    pos(0)
{}

RayBatch::iterator::iterator(RayBatch* batch, size_t pos) :
    data(batch),
    pos(pos)
{}



RayBatch::iterator& RayBatch::iterator::operator++() {
    // Increment our position
    this->pos++;
    // Return
    return *this;
}

RayBatch::iterator RayBatch::iterator::operator+(size_t n) const {
    // Increment a new position
    size_t new_pos = this->pos + 1;
    // Return
    return *this;
}

RayBatch::iterator& RayBatch::iterator::operator+=(size_t n) {
    // Increment our position
    this->pos += n;
    // Return
    return *this;
}





/***** RAYBATCH CONSTANT ITERATOR *****/
RayBatch::const_iterator::const_iterator(const RayBatch* batch) :
    data(batch),
    pos(0)
{}

RayBatch::const_iterator::const_iterator(const RayBatch* batch, size_t pos) :
    data(batch),
    pos(pos)
{}



RayBatch::const_iterator& RayBatch::const_iterator::operator++() {
    // Increment our position
    this->pos++;
    // Return
    return *this;
}

RayBatch::const_iterator RayBatch::const_iterator::operator+(size_t n) const {
    // Increment a new position
    size_t new_pos = this->pos + 1;
    // Return
    return *this;
}

RayBatch::const_iterator& RayBatch::const_iterator::operator+=(size_t n) {
    // Increment our position
    this->pos += n;
    // Return
    return *this;
}





/***** RAYBATCH *****/
HOST_DEVICE RayBatch::RayBatch(size_t x1, size_t y1, size_t x2, size_t y2) {
    
}

HOST_DEVICE RayBatch::RayBatch(const RayBatch& other) {

}

HOST_DEVICE RayBatch::RayBatch(RayBatch&& other) {

}

HOST_DEVICE RayBatch::~RayBatch() {

}



#ifdef CUDA
RayBatch* RayBatch::GPU_create(size_t x1, size_t y1, size_t x2, size_t y2, void* ptr) {

}

RayBatch* RayBatch::GPU_create(const RayBatch& other, void* ptr) {

}

RayBatch RayBatch::GPU_copy(RayBatch* ptr_gpu) {

}

void RayBatch::GPU_free(RayBatch* ptr_gpu) {

}
#endif



HOST_DEVICE Ray& RayBatch::operator[](size_t n) {

}

HOST_DEVICE Ray RayBatch::operator[](size_t n) const {

}



HOST_DEVICE RayBatch& RayBatch::operator=(RayBatch&& other) {
    
}

HOST_DEVICE void RayTracer::swap(RayBatch& rb1, RayBatch& rb2) {
    
}





/***** RAYGENERATOR CONSTANT ITERATOR *****/
RayGenerator::const_iterator::const_iterator(const RayGenerator* generator) :
    data(generator),
    pos(0)
{}

RayGenerator::const_iterator::const_iterator(const RayGenerator* generator, size_t pos) :
    data(generator),
    pos(pos)
{}



RayGenerator::const_iterator& RayGenerator::const_iterator::operator++() {
    // Increment the pos appropriately
    this->pos += this->data->batch_size;
    size_t max = this->data->width * this->data->height * this->data->n_rays;
    if (this->pos > this->data->total_rays) {
        this->pos = this->data->total_rays;
    }

    // Done, return ourselves
    return *this;
}

RayGenerator::const_iterator RayGenerator::const_iterator::operator+(size_t n) const {
    // Increment the pos appropriately
    size_t new_pos = this->pos + n * this->data->batch_size;
    if (new_pos > this->data->total_rays) {
        new_pos = this->data->total_rays;
    }

    // Done, return a new one
    return const_iterator(this->data, new_pos);
}

RayGenerator::const_iterator& RayGenerator::const_iterator::operator+=(size_t n) {
    // Increment the pos appropriately
    this->pos += n * this->data->batch_size;
    if (this->pos > this->data->total_rays) {
        this->pos = this->data->total_rays;
    }

    // Done, return ourselves
    return *this;
}





/***** RAYGENERATOR *****/
RayGenerator::RayGenerator(size_t width, size_t height, size_t n_rays, size_t batch_size) :
    width(width),
    height(height),
    n_rays(n_rays),
    total_rays(width * height * n_rays),
    batch_size(batch_size)
{}

RayGenerator::RayGenerator(const RayGenerator& other) :
    width(other.width),
    height(other.height),
    n_rays(other.n_rays),
    total_rays(width * height * n_rays),
    batch_size(other.batch_size)
{}

RayGenerator::RayGenerator(RayGenerator&& other) :
    width(other.width),
    height(other.height),
    n_rays(other.n_rays),
    total_rays(width * height * n_rays),
    batch_size(other.batch_size)
{}



RayBatch RayGenerator::operator[](size_t index) const {
    // Compute the start & end of the batch in a linear space
    size_t n1, n2;
    n1 = this->batch_size * index;
    n2 = n1 + batch_size;
    if (n1 > this->total_rays) { n1 = this->total_rays; }
    if (n2 > this->total_rays) { n2 = this->total_rays; }

    // Convert those to a 2D-space
    size_t x1, y1, x2, y2;
    x1 = n1 % this->width;
    y1 = n1 / this->width;
    x2 = n2 % this->width;
    y2 = n2 / this->width;

    // Create a RayBatch with those numbers
    return RayBatch(x1, y1, x2, y2);
}



RayGenerator& RayGenerator::operator=(RayGenerator&& other) {
    // Only swap if not the same
    if (this != &other) {
        swap(*this, other);
    }
    return *this;
}

void RayTracer::swap(RayGenerator& rc1, RayGenerator& rc2) {
    using std::swap;

    // Simply swap all the sizes
    swap(rc1.width, rc2.width);
    swap(rc1.height, rc2.height);
    swap(rc1.n_rays, rc2.n_rays);
    swap(rc1.total_rays, rc2.total_rays);
    swap(rc1.batch_size, rc2.batch_size);
}

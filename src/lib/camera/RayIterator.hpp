/* RAY ITERATOR.hpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 15:54:05
 * Last edited:
 *   13/07/2020, 14:36:55
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayIterator class is used as a wrapper around a Camera to easily
 *   iterate over Rays. Note that his will likely only be used on the CPU,
 *   and so contains no functionality to exist on the GPU whatsoever.
**/

#ifndef RAYITERATOR_HPP
#define RAYITERATOR_HPP

#include "Ray.hpp"
#include "Camera.hpp"

namespace RayTracer {
    /* Wraps a camera, and allows the user to neatly iterate over all possible rays, one at a time. */
    class RayIterator {
    private:
        /* Reference to the Camera which the Iterator wraps. */
        Camera& camera;
        /* Number of rays to cast per pixel. */
        size_t n_rays;

    public:
        /* Iterator of the RayIterator. Note that it only supports non-mutable iterators, as the Rays generated are not, in fact, stored. */
        class const_iterator {
        private:
            /* Reference to the wrapping RayIterator. */
            const RayIterator* data;
            /* Current position in the iterator. */
            size_t pos;
            /* Maximum position that the iterator never surpasses. */
            size_t max;

        public:
            /* Constructor for the const_iterator which starts at the beginning. */
            const_iterator(const RayIterator* data);
            /* Constructor for the const_iterator which starts at the given position. */
            const_iterator(const RayIterator* data, size_t pos);

            /* Returns true if this const_iterator is equal the other one. */
            inline bool operator==(const const_iterator& other) const { return this->data == other.data && this->pos == other.pos; }
            /* Returns true if this const_iterator is not equal the other one. */
            inline bool operator!=(const const_iterator& other) const { return this->data != other.data || this->pos != other.pos; }
            /* Compares to see if this iterator is before the given iterator. If they do not point to the same data, returns false. */
            inline bool operator<(const const_iterator& other) const { return this->data == other.data && this->pos < other.pos; }
            /* Compares to see if this iterator is before or equal to the given iterator. If they do not point to the same data, returns false. */
            inline bool operator<=(const const_iterator& other) const { return this->data == other.data && this->pos <= other.pos; }
            /* Compares to see if this iterator is after the given iterator. If they do not point to the same data, returns false. */
            inline bool operator>(const const_iterator& other) const { return this->data == other.data && this->pos > other.pos; }
            /* Compares to see if this iterator is after or equal to the given iterator. If they do not point to the same data, returns false. */
            inline bool operator>=(const const_iterator& other) const { return this->data == other.data && this->pos >= other.pos; }

            /* Increments the const_iterator to the next position (inplace). */
            const_iterator& operator++();
            /* Increments the const_iterator with n positions (creates new iterator). */
            const_iterator operator+(size_t n) const;
            /* Increments the const_iterator with n positions (inplace). */
            const_iterator& operator+=(size_t n);

            /* De-references the iterator, returning a Ray. */
            inline Ray operator*() const { return (*(this->data))[this->pos]; }
        };

        /* Constructor for the RayIterator class, which simply takes a camera and the number of rays to cast per pixel. */
        RayIterator(Camera& camera, size_t n_rays);
        /* Copy constructor for the RayIterator class. */
        RayIterator(const RayIterator& other);
        /* Move constructor for the RayIterator class. */
        RayIterator(RayIterator&& other);

        /* Generate a Ray with given index as seen from all possible rays in the Frame. */
        inline Ray operator[](size_t n) const { return this->camera.cast(n % (this->camera.frame_width * this->n_rays), n / (this->camera.frame_width * this->n_rays)); }
        /* Generate a Ray at given coordinate in the Frame. */
        inline Ray operator[](const Point2& coord) const { return this->camera.cast(coord); }

        /* Copy assignment operator for the RayIterator class. */
        inline RayIterator& operator=(const RayIterator& other) { return *this = RayIterator(other); }
        /* Move assignment operator for the RayIterator class. */
        RayIterator& operator=(RayIterator&& other);
        /* Swap operator for the RayIterator class. */
        friend void swap(RayIterator& ri1, RayIterator& ri2);
        
        /* Acquire an iterator to the first ray. */
        inline const_iterator begin() const { return RayIterator::const_iterator(this); }
        /* Acquire an iterator to beyond the last ray. */
        inline const_iterator end() const { return RayIterator::const_iterator(this, this->camera.frame_width * this->camera.frame_height * this->n_rays); }
    };

    /* Swap operator for the RayIterator class. */
    void swap(RayIterator& ri1, RayIterator& ri2);
}

#endif

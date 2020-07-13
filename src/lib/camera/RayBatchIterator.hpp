/* RAY BATCH ITERATOR.hpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:21:18
 * Last edited:
 *   13/07/2020, 15:27:46
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayBatchIterator class is used as a wrapper around a Camera to easily
 *   iterate over Rays. In contrast to the normal RayIterator, does this
 *   iterator return a bunch of rays (a RayBatch) to facilitate massive
 *   parallelism (a.k.a., the GPU) over a portion of all available rays, to
 *   reduce memory usage.
**/

#ifndef RAYBATCHITERATOR_HPP
#define RAYBATCHITERATOR_HPP

#include "Ray.hpp"
#include "Camera.hpp"

#include <iostream>

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    /* Represents more than one Ray, and is GPU-ready. */
    class RayBatch {
    private:
        /* A reference to the list of Rays. */
        Ray* rays;
        /* The number of rays represented by this RayBatch. */
        size_t n_rays;
        /* The start index of the Rays represented by the RayBatch. */
        size_t start;
        /* The stop index of the Rays represented by the RayBatch. */
        size_t stop;
        /* The width of the to-be-rendered Frame. */
        size_t width;
        /* The height of the to-be-rendered Frame. */
        size_t height;
        /* The number of rays per pixel. */
        size_t rays_per_pixel;
    
    public:
        /* Iterator over the RayBatch class, providing mutable access. */
        class iterator {
        private:
            /* Reference to the wrapped data pointer. */
            RayBatch* data;
            /* The current position in the iterator. */
            size_t pos;
            /* Maximum position that the iterator never surpasses. */
            size_t max;
        
        public:
            /* Constructor for the iterator class, which takes the to-be-iterated-over Raybatch object and returns an iterator at the beginning. */
            HOST_DEVICE iterator(RayBatch* batch);
            /* Constructor for the iterator class, which takes the to-be-iterated-over Raybatch object and an index to begin on. */
            HOST_DEVICE iterator(RayBatch* batch, size_t pos);

            /* Returns true if this iterator is equal the other one. */
            HOST_DEVICE inline bool operator==(const iterator& other) const { return this->data == other.data && this->pos == other.pos; }
            /* Returns true if this iterator is not equal the other one. */
            HOST_DEVICE inline bool operator!=(const iterator& other) const { return this->data != other.data || this->pos != other.pos; }
            /* Compares to see if this iterator is before the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator<(const iterator& other) const { return this->data == other.data && this->pos < other.pos; }
            /* Compares to see if this iterator is before or equal to the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator<=(const iterator& other) const { return this->data == other.data && this->pos <= other.pos; }
            /* Compares to see if this iterator is after the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator>(const iterator& other) const { return this->data == other.data && this->pos > other.pos; }
            /* Compares to see if this iterator is after or equal to the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator>=(const iterator& other) const { return this->data == other.data && this->pos >= other.pos; }

            /* Increments the const_iterator to the next position (inplace). */
            HOST_DEVICE iterator& operator++();
            /* Increments the const_iterator with n positions (creates new iterator). */
            HOST_DEVICE iterator operator+(size_t n) const;
            /* Increments the const_iterator with n positions (inplace). */
            HOST_DEVICE iterator& operator+=(size_t n);

            /* De-references the iterator, returning an Ray. */
            HOST_DEVICE inline Ray& operator*() { return (*(this->data))[this->pos]; }
        };

        /* Iterator over the RayBatch class, providing non-mutable access. */
        class const_iterator {
        private:
            /* Reference to the wrapped data pointer. */
            const RayBatch* data;
            /* The current position in the iterator. */
            size_t pos;
            /* Maximum position that the iterator never surpasses. */
            size_t max;
        
        public:
            /* Constructor for the const_iterator class, which takes the to-be-iterated-over Raybatch object and returns an iterator at the beginning. */
            HOST_DEVICE const_iterator(const RayBatch* batch);
            /* Constructor for the const_iterator class, which takes the to-be-iterated-over Raybatch object and an index to begin on. */
            HOST_DEVICE const_iterator(const RayBatch* batch, size_t pos);

            /* Returns true if this iterator is equal the other one. */
            HOST_DEVICE inline bool operator==(const const_iterator& other) const { return this->data == other.data && this->pos == other.pos; }
            /* Returns true if this iterator is not equal the other one. */
            HOST_DEVICE inline bool operator!=(const const_iterator& other) const { return this->data != other.data || this->pos != other.pos; }
            /* Compares to see if this iterator is before the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator<(const const_iterator& other) const { return this->data == other.data && this->pos < other.pos; }
            /* Compares to see if this iterator is before or equal to the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator<=(const const_iterator& other) const { return this->data == other.data && this->pos <= other.pos; }
            /* Compares to see if this iterator is after the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator>(const const_iterator& other) const { return this->data == other.data && this->pos > other.pos; }
            /* Compares to see if this iterator is after or equal to the given iterator. If they do not point to the same data, returns false. */
            HOST_DEVICE inline bool operator>=(const const_iterator& other) const { return this->data == other.data && this->pos >= other.pos; }

            /* Increments the const_iterator to the next position (inplace). */
            HOST_DEVICE const_iterator& operator++();
            /* Increments the const_iterator with n positions (creates new iterator). */
            HOST_DEVICE const_iterator operator+(size_t n) const;
            /* Increments the const_iterator with n positions (inplace). */
            HOST_DEVICE const_iterator& operator+=(size_t n);

            /* De-references the iterator, returning an Ray. */
            HOST_DEVICE inline Ray operator*() const { return (*(this->data))[this->pos]; }
        };

        /* Constructor for the RayBatch, which takes a Camera to create the Rays with, the number of rays per pixel, the start index in the total amount of rays represented by the camera & number of rays per pixel and the stop index in the total amount of rays. Note that the start is inclusive, but the stop isn't. */
        HOST_DEVICE RayBatch(const Camera& camera, size_t n_rays, size_t start, size_t stop);
        /* Copy constructor for the RayBatch class. */
        HOST_DEVICE RayBatch(const RayBatch& other);
        /* Move constructor for the RayBatch class. */
        HOST_DEVICE RayBatch(RayBatch&& other);
        /* Deconstructor for the RayBatch class. */
        HOST_DEVICE ~RayBatch();

        #ifdef CUDA
        /* CPU-side constructor for a GPU-side RayBatch object. Takes a Camera to create the Rays with, the number of rays per pixel, the start index in the total amount of rays represented by the camera & number of rays per pixel and the stop index in the total amount of rays. Note that the start is inclusive, but the stop isn't. If ptr == nullptr, does not allocate but instead only copies to the given memory location. */
        static RayBatch* GPU_create(const Camera& camera, size_t n_rays, size_t start, size_t stop, void* ptr = nullptr);
        /* CPU-side constructor for a GPU-side RayBatch object. Takes another RayBatch and copies it. If ptr == nullptr, does not allocate but instead only copies to the given memory location. */
        static RayBatch* GPU_create(const RayBatch& other, void* ptr = nullptr);
        /* Copies a GPU-side RayBatch object back to the CPU as a stack-allocated new object. */
        static RayBatch GPU_copy(RayBatch* ptr_gpu);
        /* CPU-side deconstructor for a GPU-side RayBatch object. */
        static void GPU_free(RayBatch* ptr_gpu);
        #endif

        /* Allows (non-mutable) access to the internal Rays. Note that no out-of-bounds checking is performed, and so undefined behaviour may occur. */
        HOST_DEVICE inline Ray operator[](size_t n) const { return this->rays[n]; }
        /* Allows (mutable) access to the internal Rays. Note that no out-of-bounds checking is performed, and so undefined behaviour may occur. */
        HOST_DEVICE inline Ray& operator[](size_t n) { return this->rays[n]; }

        /* Copy assignment operator for the RayBatch class. */
        HOST_DEVICE inline RayBatch& operator=(const RayBatch& other) { return *this = RayBatch(other); }
        /* Move assignment operator for the RayBatch class. */
        HOST_DEVICE RayBatch& operator=(RayBatch&& other);
        /* Swap operator for the RayBatch class. */
        friend HOST_DEVICE void swap(RayBatch& rb1, RayBatch& rb2);

        /* Returns a mutable iterator which points to the start of all available Rays. */
        HOST_DEVICE inline iterator begin() { return RayBatch::iterator(this); }
        /* Returns a non-mutable iterator which points to the start of all available Rays. */
        HOST_DEVICE inline const_iterator begin() const { return RayBatch::const_iterator(this); }
        /* Returns a mutable iterator which points beyond the end of all available Rays. */
        HOST_DEVICE inline iterator end() { return RayBatch::iterator(this, this->n_rays); }
        /* Returns a non-mutable iterator which points beyond the end of all available Rays. */
        HOST_DEVICE inline const_iterator end() const { return RayBatch::const_iterator(this, this->n_rays); }
    };

    /* Swap operator for the RayBatch class. */
    HOST_DEVICE void swap(RayBatch& rb1, RayBatch& rb2);

    /* Wraps a camera, and allows the user to neatly iterate over all possible rays, multiple ones at a time. */
    class RayBatchIterator {
    private:
        /* Reference to the Camera which the Iterator wraps. */
        Camera& camera;
        /* Number of rays to cast per pixel. */
        size_t n_rays;
        /* Number of rays per batch. */
        size_t batch_size;

    public:
        /* Iterator of the RayIterator. Note that it only supports non-mutable iterators, as the Rays generated are not, in fact, stored. */
        class const_iterator {
        private:
            /* Reference to the wrapping RayIterator. */
            const RayBatchIterator* data;
            /* Current position in the iterator. */
            size_t pos;
            /* Maximum position that the iterator never surpasses. */
            size_t max;

        public:
            /* Constructor for the const_iterator which starts at the beginning. */
            const_iterator(const RayBatchIterator* data);
            /* Constructor for the const_iterator which starts at the given position. */
            const_iterator(const RayBatchIterator* data, size_t pos);

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

            /* De-references the iterator, returning a RayBatch. */
            inline RayBatch operator*() const { return (*(this->data))[this->pos]; }
        };

        /* Constructor for the RayIterator class, which simply takes a camera and the number of rays to cast per pixel. Finally, the batch size can also be tweaked. */
        RayBatchIterator(Camera& camera, size_t n_rays, size_t batch_size = 500000);
        /* Copy constructor for the RayIterator class. */
        RayBatchIterator(const RayBatchIterator& other);
        /* Move constructor for the RayIterator class. */
        RayBatchIterator(RayBatchIterator&& other);

        /* Generate a RayBatch with given index as seen from all possible rays in the Frame. */
        RayBatch operator[](size_t n) const;

        /* Copy assignment operator for the RayIterator class. */
        inline RayBatchIterator& operator=(const RayBatchIterator& other) { return *this = RayBatchIterator(other); }
        /* Move assignment operator for the RayIterator class. */
        RayBatchIterator& operator=(RayBatchIterator&& other);
        /* Swap operator for the RayIterator class. */
        friend void swap(RayBatchIterator& ri1, RayBatchIterator& ri2);
        
        /* Acquire an iterator to the first ray. */
        inline const_iterator begin() const { return RayBatchIterator::const_iterator(this); }
        /* Acquire an iterator to beyond the last ray. */
        inline const_iterator end() const { return RayBatchIterator::const_iterator(this, this->camera.frame_width * this->camera.frame_height * this->n_rays); }
    };

    /* Swap operator for the RayIterator class. */
    void swap(RayBatchIterator& ri1, RayBatchIterator& ri2);
}

#endif

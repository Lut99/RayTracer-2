/* RAY GENERATOR.hpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:46:28
 * Last edited:
 *   08/07/2020, 22:12:06
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

#ifndef RAYGENERATOR_HPP
#define RAYGENERATOR_HPP

#include <cstddef>

#include "Ray.hpp"

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    /* Allows the user to generate new batches of Rays, which can then be processed in a parallel fashion. Note that this one does copy to the GPU. */
    class RayBatch {
    public:
        /* Mutable iterator over the Rays in this RayBatch. */
        class iterator {
        private:
            /* Index of the last-accessed element. */
            size_t pos;
            /* Reference to the internal RayBatch. */
            RayBatch* data;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            HOST_DEVICE iterator(RayBatch* batch);
            /* Constructor for the iterator class, starting at a specified beginning. */
            HOST_DEVICE iterator(RayBatch* batch, size_t pos);

            /* Checks if two const_iterator are equal. */
            HOST_DEVICE inline bool operator==(const iterator& other) { return this->pos == other.pos && this->data == other.data; }
            /* Checks if two const_iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const iterator& other) { return this->pos != other.pos || this->data != other.data; }

            /* Increments the const_iterator to the next position (in the x-axis). */
            HOST_DEVICE iterator& operator++();
            /* Moves the const_iterator n batches forward. */
            HOST_DEVICE iterator operator+(size_t n) const;
            /* Moves the const_iterator n batches forward. */
            HOST_DEVICE iterator& operator+=(size_t n);

            /* Dereferences the const_iterator. */
            inline HOST_DEVICE Ray& operator*() const { return this->data->operator[](this->pos); }
        };

        /* Non-mutable iterator of the Rays in this RayBatch. */
        class const_iterator {
        private:
            /* Index of the last-accessed element. */
            size_t pos;
            /* Reference to the internal RayBatch. */
            const RayBatch* data;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            HOST_DEVICE const_iterator(const RayBatch* batch);
            /* Constructor for the iterator class, starting at a specified beginning. */
            HOST_DEVICE const_iterator(const RayBatch* batch, size_t pos);

            /* Checks if two const_iterator are equal. */
            HOST_DEVICE inline bool operator==(const const_iterator& other) { return this->pos == other.pos && this->data == other.data; }
            /* Checks if two const_iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const const_iterator& other) { return this->pos != other.pos || this->data != other.data; }

            /* Increments the const_iterator to the next position (in the x-axis). */
            HOST_DEVICE const_iterator& operator++();
            /* Moves the const_iterator n batches forward. */
            HOST_DEVICE const_iterator operator+(size_t n) const;
            /* Moves the const_iterator n batches forward. */
            HOST_DEVICE const_iterator& operator+=(size_t n);

            /* Dereferences the const_iterator. */
            inline HOST_DEVICE Ray operator*() const { return this->data->operator[](this->pos); }
        };

        /* Constructor for the RayBatch which takes an area (start position and stop position) in a 2D-frame which will be rendered. */
        HOST_DEVICE RayBatch(size_t x1, size_t y1, size_t x2, size_t y2);
        /* Copy constructor for the RayBatch. */
        HOST_DEVICE RayBatch(const RayBatch& other);
        /* Move constructor for the RayBatch. */
        HOST_DEVICE RayBatch(RayBatch&& other);
        /* Deconstructor for the RayBatch class. */
        HOST_DEVICE ~RayBatch();

        #ifdef CUDA
        /* CPU-side constructor for a GPU-side RayBatch which takes an area (start position and stop position) in a 2D-frame which will be rendered. If ptr == nullptr, does not allocate but only copies to that memory location. */
        static RayBatch* GPU_create(size_t x1, size_t y1, size_t x2, size_t y2, void* ptr = nullptr);
        /* CPU-side constructor for a GPU-side RayBatch which copies another RayBatch. If ptr == nullptr, does not allocate but only copies to that memory location. */
        static RayBatch* GPU_create(const RayBatch& other, void* ptr = nullptr);
        /* Copies a GPU-side RayBatch to the CPU. */
        static RayBatch GPU_copy(RayBatch* ptr_gpu);
        /* CPU-side deconstructor for a GPU-side RayBatch. */
        static void GPU_free(RayBatch* ptr_gpu);
        #endif

        /* Allows (mutable) access to a Ray in the RayBatch. */
        HOST_DEVICE Ray& operator[](size_t n);
        /* Allows (non-mutable) access to a Ray in the RayBatch. */
        HOST_DEVICE Ray operator[](size_t n) const;

        /* Copy assignment operator for the RayBatch class. */
        HOST_DEVICE inline RayBatch& operator=(const RayBatch& other) { return *this = RayBatch(other); }
        /* Move assignment operator for the RayBatch class. */
        HOST_DEVICE RayBatch& operator=(RayBatch&& other);
        /* Swap operator for the RayBatch class. */
        friend HOST_DEVICE void swap(RayBatch& rb1, RayBatch& rb2);
    };
    
    /* Swap operator for the RayBatch class. */
    HOST_DEVICE void swap(RayBatch& rb1, RayBatch& rb2);

    /* Generates RayBatches step-by-step based on a single frame. */
    class RayGenerator {
    private:
        /* The width (in pixels) of the frame we want to render. */
        size_t width;
        /* The height (in pixels) of the frame we want to render. */
        size_t height;
        /* The number of rays we will cast per pixel. */
        size_t n_rays;
        /* The total number of rays that will be case per frame. */
        size_t total_rays;
        /* The size of each RayBatch generated. */
        size_t batch_size;

    public:
        /* Non-mutable iterator for the RayGenerator class (since the RayGenerator is non-mutable by default). */
        class const_iterator {
        private:
            /* Index of the last-used element. */
            size_t pos;
            /* Reference to the internal RayGenerator. */
            const RayGenerator* data;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            const_iterator(const RayGenerator* generator);
            /* Constructor for the iterator class, starting at a specified beginning. */
            const_iterator(const RayGenerator* generator, size_t pos);

            /* Checks if two const_iterator are equal. */
            inline bool operator==(const const_iterator& other) { return this->pos == other.pos && this->data == other.data; }
            /* Checks if two const_iterator are unequal. */
            inline bool operator!=(const const_iterator& other) { return this->pos != other.pos || this->data != other.data; }

            /* Increments the const_iterator to the next position (in the x-axis). */
            const_iterator& operator++();
            /* Moves the const_iterator n batches forward. */
            const_iterator operator+(size_t n) const;
            /* Moves the const_iterator n batches forward. */
            const_iterator& operator+=(size_t n);

            /* Dereferences the const_iterator. */
            inline RayBatch operator*() const { return this->data->operator[](this->pos); }
        };

        /* Constructor for the RayGenerator class, which takes the dimensions of the frame and the number of rays we cast per pixel. */
        RayGenerator(size_t width, size_t height, size_t n_rays, size_t batch_size);
        /* Copy constructor for the RayGenerator class. */
        RayGenerator(const RayGenerator& other);
        /* Move constructor for the RayGenerator class. */
        RayGenerator(RayGenerator&& other);

        /* Accesses a (new) batch at the given position. */
        RayBatch operator[](size_t index) const;

        /* Copy assignment operator for the RayGenerator class. */
        inline RayGenerator& operator=(const RayGenerator& other) { return *this = RayGenerator(other); }
        /* Move assignment operator for the RayGenerator class. */
        RayGenerator& operator=(RayGenerator&& other);
        /* Swap operator for the RayGenerator class. */
        friend void swap(RayGenerator& rc1, RayGenerator& rc2);

        /* Returns a const_iterator to the beginning of this RayGenerator. */
        inline const_iterator begin() const { return RayGenerator::const_iterator(this); };
        /* Returns a const_iterator beyond the end of this RayGenerator. */
        inline const_iterator end() const { return RayGenerator::const_iterator(this, this->width * this->height * this->n_rays); }
    };

    /* Swap operator for the RayGenerator class. */
    void swap(RayGenerator& rc1, RayGenerator& rc2);
}

#endif

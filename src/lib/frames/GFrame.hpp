/* GFRAME.hpp
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 3:13:25 PM
 * Last edited:
 *   7/4/2020, 4:12:52 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The GFrame-class is the GPU-side counterpart of the Frame class. It
 *   specializes in copying the data to and from the GPU and in accessing
 *   operations on the GPU - but not in writing to a PNG, as the GPU has no
 *   file access anyway.
**/

#ifndef GFRAME_HPP
#define GFRAME_HPP

#include "Frame.hpp"

namespace RayTracer {
    /* A struct which wraps a Pixel, i.e., three consecutive doubles. */
    class GFrame;
    struct GPixel {
    private:
        /* Pointer to the data in the GFrame class. */
        double* data;
        /* Position of the GPixel in the parent GFrame. */
        Coordinate pos;

        /* Constructor for the GPixel struct, only accessible from the GFrame class. */
        __device__ GPixel(const Coordinate& pos, double* const data);
    
    public:
        /* Reference to the red-channel value. */
        double& r;
        /* Reference to the green-channel value. */
        double& g;
        /* Reference to the blue-channel value. */
        double& b;

        /* Copy constructor for the GPixel class. */
        __device__ GPixel(const GPixel& other);
        /* Move constructor for the GPixel class. */
        __device__ GPixel(GPixel&& other);

        /* Allows the GPixel to be indexed numerically (non-mutable). */
        __device__ double operator[](const size_t i) const;
        /* Allows the GPixel to be indexed numerically (mutable). */
        __device__ double& operator[](const size_t i);

        /* Returns the x-position of the GPixel. */
        __device__ inline size_t x() const { return this->pos.x; }
        /* Returns the y-position of the GPixel. */
        __device__ inline size_t y() const { return this->pos.y; }

        /* Mark the GFrame class as friend. */
        friend class GFrame;
    };

    /* A class which is used to manipulate the GPixels of an output GFrame. Can also write its internal buffer to a given file as PNG. */
    class GFrame {
    private:
        /* The data containing all pixel values. Is height * width * 3 large. Can either refer to own data or GPU-located, external data. */
        void* data;

    public:
        /* Width of the GFrame (in pixels). */
        const size_t width;
        /* Height of the GFrame (in pixels). */
        const size_t height;
        /* Pitch size of the CUDA-allocated memory area. */
        size_t pitch;

        /* Mutable iterator for the GFrame class. */
        class iterator {
        private:
            /* Index of the last-used element. */
            Coordinate pos;
            /* Width of the parent GFrame. */
            size_t width;
            /* Height of the parent GFrame. */
            size_t height;
            /* Reference to the internal GFrame. */
            GFrame* data;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            __device__ iterator(GFrame* frame);
            /* Constructor for the iterator class, starting at a specified beginning. */
            __device__ iterator(GFrame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            __device__ inline bool operator==(const iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            __device__ inline bool operator!=(const iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            __device__ iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            __device__ iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            __device__ iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            __device__ iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            __device__ iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            __device__ inline GPixel operator*() const { return this->data->operator[]({this->pos.x, this->pos.y}); }
        };

        /* Non-mutable iterator for the GFrame class. */
        class const_iterator {
        private:
            /* Index of the last-used element. */
            Coordinate pos;
            /* Width of the parent GFrame. */
            size_t width;
            /* Height of the parent GFrame. */
            size_t height;
            /* Reference to the internal GFrame. */
            const GFrame* data;

        public:
            /* Constructor for the GFrame class, starting at the beginning (0, 0). */
            __device__ const_iterator(const GFrame* frame);
            /* Constructor for the GFrame class, starting at a specified beginning. */
            __device__ const_iterator(const GFrame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            __device__ inline bool operator==(const const_iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            __device__ inline bool operator!=(const const_iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            __device__ const_iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            __device__ const_iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            __device__ const_iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            __device__ const_iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            __device__ const_iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            __device__ inline const GPixel operator*() const { return this->data->operator[]({this->pos.x, this->pos.y}); }
        };

        /* Constructor for the GFrame class which takes a GPU-pointer and uses that to initialize. */
        __device__ GFrame(cudaPitchedPtr ptr);

        /* Indexes the GFrame for a given coordinate (non-mutable). */
        __device__ const GPixel operator[](const Coordinate& index) const;
        /* Indexes the GFrame for a given coordinate (mutable). */
        __device__ GPixel operator[](const Coordinate& index);

        /* Copies a given CPU-side Frame to the GPU. */
        __host__ static cudaPitchedPtr toGPU(const Frame& frame);
        /* Retrieve a GPU-side Frame as a CPU-side Frame. Also frees the GPU-side memory. */
        __host__ static Frame fromGPU(cudaPitchedPtr ptr);

        /* Copy assignment operator for the GFrame class. Note that this throws a runtime_error if the GFrame is not the same size. */
        __device__ GFrame& operator=(GFrame other);
        /* Move assignment operator for the GFrame class. Note that this throws a runtime_error if the GFrame is not the same size. */
        __device__ GFrame& operator=(GFrame&& other);
        /* Swap operator for the GFrame class. Note that this does not swap the size, and therefore expects the two GFrames to have the same size. */
        __device__ friend void swap(GFrame& f1, GFrame& f2);

        /* Returns a mutable iterator to the beginning of this GFrame. */
        __device__ inline iterator begin() { return GFrame::iterator(this); }
        /* Returns a mutable iterator beyond the end of this GFrame. */
        __device__ inline iterator end() { return GFrame::iterator(this, {0, this->height}); }
        /* Returns a const_iterator to the beginning of this GFrame. */
        __device__ inline const_iterator begin() const { return GFrame::const_iterator(this); };
        /* Returns a const_iterator beyond the end of this GFrame. */
        __device__ inline const_iterator end() const { return GFrame::const_iterator(this, {0, this->height}); }
    };

    __device__ void swap(GFrame& f1, GFrame& f2);
}

#endif

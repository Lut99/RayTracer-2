/* FRAME.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:24 PM
 * Last edited:
 *   13/07/2020, 13:39:37
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Frame class is a container for the resulting pixels which can be
 *   relatively easily written to PNG. In fact, the writing to a PNG-file
 *   is also supported.
**/

#ifndef FRAME_HPP
#define FRAME_HPP

#include <cstddef>
#include <ostream>
#include <string>

#include "GPUDev.hpp"

#include "PixelCoord.hpp"


namespace RayTracer {
    /* A struct which wraps a Pixel, i.e., three consecutive doubles. */
    class Frame;
    struct Pixel {
    private:
        /* Pointer to the data in the Frame class. */
        double* data;
        /* Determines if we're using local values or external values. */
        bool is_external;
        /* Position of the Pixel in the parent Frame. */
        PixelCoord frame_pos;

        /* Locally-stored red-value, only used when is_external = false. */
        double local_r;
        /* Locally-stored green-value, only used when is_external = false. */
        double local_g;
        /* Locally-stored blue-value, only used when is_external = false. */
        double local_b;

        /* Constructor for the Pixel struct, only accessible from the Frame class. */
        HOST_DEVICE Pixel(const PixelCoord& pos, double* const data);
    
    public:
        /* Reference to the red-channel value. */
        double& r;
        /* Reference to the green-channel value. */
        double& g;
        /* Reference to the blue-channel value. */
        double& b;

        /* Constructor for the Pixel class, which defines it locally with an r, g and b. */
        HOST_DEVICE Pixel(double r, double g, double b);
        /* Copy constructor for the Pixel class. */
        HOST_DEVICE Pixel(const Pixel& other);
        /* Move constructor for the Pixel class. */
        HOST_DEVICE Pixel(Pixel&& other);

        /* Allows the pixel to be compared to another pixel. */
        HOST_DEVICE inline bool operator==(const Pixel& other) const { return this->r == other.r && this->g == other.g && this->b == other.b; }
        /* Allows the pixel to be compared to another pixel by inequality. */
        HOST_DEVICE inline bool operator!=(const Pixel& other) const { return this->r != other.r || this->g != other.g || this->b != other.b; }
        /* Allows the pixel to be compared to see if all values are less than given constant. */
        HOST_DEVICE inline bool operator<(double c) const { return this->r < c && this->g < c && this->b < c; }
        /* Allows the pixel to be compared to see if all values are less or equal than given constant. */
        HOST_DEVICE inline bool operator<=(double c) const { return this->r <= c && this->g <= c && this->b <= c; }

        /* Allos the pixel to be subtracted from another pixel. */
        HOST_DEVICE inline Pixel operator-(const Pixel& other) const { return Pixel(this->r - other.r, this->g - other.g, this->b - other.b); }

        /* Allows the pixel to be indexed numerically (non-mutable). */
        HOST_DEVICE double operator[](const size_t i) const;
        /* Allows the pixel to be indexed numerically (mutable). */
        HOST_DEVICE double& operator[](const size_t i);

        /* Returns the position of the Pixel in the frame. */
        HOST_DEVICE inline PixelCoord pos() const { return this->pos; }
        /* Returns the x-position of the Pixel. */
        HOST_DEVICE inline pixel_coord x() const { return this->pos.x; }
        /* Returns the y-position of the Pixel. */
        HOST_DEVICE inline pixel_coord y() const { return this->pos.y; }

        /* Copy assignment operator for the Pixel class. */
        HOST_DEVICE inline Pixel& operator=(const Pixel& other) { return *this = Pixel(other); }
        /* Move assignment operator for the Pixel class. */
        HOST_DEVICE Pixel& operator=(Pixel&& other);
        /* Swap operator for the Pixel class. */
        friend HOST_DEVICE void swap(Pixel& p1, Pixel& p2);

        /* Allows the Pixel to be printed to a stream. */
        friend std::ostream& operator<<(std::ostream& os, const Pixel& pixel);

        /* Mark the Frame class as friend. */
        friend class Frame;
    };

    /* Allows the Pixel to be printed to a stream. */
    std::ostream& operator<<(std::ostream& os, const Pixel& pixel);
    /* Swap operator for the Pixel class. */
    HOST_DEVICE void swap(Pixel& p1, Pixel& p2);

    /* A class which is used to manipulate the Pixels of an output frame. Can also write its internal buffer to a given file as PNG. */
    class Frame {
    private:
        /* The data containing all pixel values. Is a flattened 2D-array, where each row has 3 * width doubles and there are height rows. */
        void* data;
        /* Stores the size of a single row in width. On the CPU, this is simply the size of a double times the number of elements per row, but this might differ on the GPU. */
        size_t pitch;
        /* Determines if the Frame uses memory which is managed by itself or by an external something. */
        bool is_external;
        
        /* Constructor for the Frame class which takes a pointer instead of allocating a new one. Note that the given pointer will still be deallocated by the class. */
        Frame(size_t width, size_t height, void* data, size_t pitch);

    public:
        /* Width of the Frame (in pixels). */
        const size_t width;
        /* Height of the Frame (in pixels). */
        const size_t height;

        /* Mutable iterator for the Frame class. */
        class iterator {
        private:
            /* Reference to the internal frame. */
            Frame* data;
            /* Index of the last-used element. */
            PixelCoord pos;
            /* Maximum index of the last-used element. */
            PixelCoord max;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            HOST_DEVICE iterator(Frame* frame);
            /* Constructor for the iterator class, starting at a specified beginning. */
            HOST_DEVICE iterator(Frame* frame, const PixelCoord& pos);

            /* Checks if two iterators are equal. */
            HOST_DEVICE inline bool operator==(const iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            HOST_DEVICE iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE iterator operator+(const PixelCoord& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE iterator& operator+=(const PixelCoord& n);

            /* Dereferences the iterator. */
            HOST_DEVICE inline Pixel operator*() const { return (*(this->data))[PixelCoord(this->pos.x, this->pos.y)]; }
        };

        /* Non-mutable iterator for the Frame class. */
        class const_iterator {
        private:
            /* Reference to the internal frame. */
            const Frame* data;
            /* Index of the last-used element. */
            PixelCoord pos;
            /* Maximum index of the last-used element. */
            PixelCoord max;

        public:
            /* Constructor for the Frame class, starting at the beginning (0, 0). */
            HOST_DEVICE const_iterator(const Frame* frame);
            /* Constructor for the Frame class, starting at a specified beginning. */
            HOST_DEVICE const_iterator(const Frame* frame, const PixelCoord& pos);

            /* Checks if two iterators are equal. */
            HOST_DEVICE inline bool operator==(const const_iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const const_iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            HOST_DEVICE const_iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE const_iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE const_iterator operator+(const PixelCoord& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE const_iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE const_iterator& operator+=(const PixelCoord& n);

            /* Dereferences the iterator. */
            HOST_DEVICE inline const Pixel operator*() const { return (*(this->data))[PixelCoord(this->pos.x, this->pos.y)]; }
        };

        /* Constructor for the Frame class. Initializes an image with the specified dimensions with uninitialized pixels. */
        Frame(size_t width, size_t height);
        /* Copy constructor for the Frame class. */
        HOST_DEVICE Frame(const Frame& other);
        /* Move constructor for the Frame class. */
        HOST_DEVICE Frame(Frame&& other);
        /* Deconstructor for the Frame class (Host-only). */
        HOST_DEVICE ~Frame();

        #ifdef CUDA
        /* CPU-side constructor for a GPU-side Frame. Allocates only if ptr == nullptr, and then copies a frame with given size to the target location. */
        static Frame* GPU_create(size_t width, size_t height, void* ptr = nullptr);
        /* CPU-side constructor for a GPU-side Frane. Allocates only if ptr == nullptr, and then copies a copied Frane to that memory location. */
        static Frame* GPU_create(const Frame& other, void* ptr = nullptr);
        /* Copies a GPU-side Frame to a newly (stack-)allocated CPU-side Frame. Does not deallocate the GPU-side. */
        static Frame GPU_copy(Frame* ptr_gpu);
        /* GPU-side destructor for the GPU-side Frame. */
        static void GPU_free(Frame* ptr_gpu);
        #endif

        /* Indexes the Frame for a given coordinate (non-mutable). */
        HOST_DEVICE const Pixel operator[](const PixelCoord& index) const;
        /* Indexes the Frame for a given coordinate (mutable). */
        HOST_DEVICE Pixel operator[](const PixelCoord& index);

        /* Writes the Frame to a PNG of our choosing. */
        void toPNG(const std::string& path) const;

        /* Copy assignment operator for the Frame class. Note that this throws a runtime_error if the Frame is not the same size. */
        HOST_DEVICE Frame& operator=(const Frame& other);
        /* Move assignment operator for the Frame class. Note that this throws a runtime_error if the Frame is not the same size. */
        HOST_DEVICE Frame& operator=(Frame&& other);
        /* Swap operator for the Frame class. Note that this does not swap the size, and therefore expects the two Frames to have the same size. */
        friend HOST_DEVICE void swap(Frame& f1, Frame& f2);

        /* Returns a mutable iterator to the beginning of this Frame. */
        HOST_DEVICE inline iterator begin() { return Frame::iterator(this); }
        /* Returns a mutable iterator beyond the end of this Frame. */
        HOST_DEVICE inline iterator end() { return Frame::iterator(this, {0, this->height}); }
        /* Returns a const_iterator to the beginning of this Frame. */
        HOST_DEVICE inline const_iterator begin() const { return Frame::const_iterator(this); };
        /* Returns a const_iterator beyond the end of this Frame. */
        HOST_DEVICE inline const_iterator end() const { return Frame::const_iterator(this, {0, this->height}); }
    };

    HOST_DEVICE void swap(Frame& f1, Frame& f2);
}

#endif

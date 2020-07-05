/* FRAME.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:24 PM
 * Last edited:
 *   05/07/2020, 17:19:36
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
#include <string>

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    #ifdef CUDA
    /* A pointer struct to facilitate moving the Frame between the CPU and the GPU. */
    struct FramePtr {
        /* Logical width of the frame. */
        size_t width;
        /* Logical height of the frame. */
        size_t height;
        /* Pitch size (in bytes) of the allocated data. */
        size_t pitch;
        /* Allocated data. */
        void* data;
    };
    #endif

    /* The Coordinate struct. */
    struct Coordinate {
        /* The target x-location. */
        size_t x;
        /* The target y-location. */
        size_t y;
    };
    
    /* A struct which wraps a Pixel, i.e., three consecutive doubles. */
    class Frame;
    struct Pixel {
    private:
        /* Pointer to the data in the Frame class. */
        double* data;
        /* Position of the Pixel in the parent Frame. */
        Coordinate pos;

        /* Constructor for the Pixel struct, only accessible from the Frame class. */
        HOST_DEVICE Pixel(const Coordinate& pos, double* const data);
    
    public:
        /* Reference to the red-channel value. */
        double& r;
        /* Reference to the green-channel value. */
        double& g;
        /* Reference to the blue-channel value. */
        double& b;

        /* Copy constructor for the Pixel class. */
        HOST_DEVICE Pixel(const Pixel& other);
        /* Move constructor for the Pixel class. */
        HOST_DEVICE Pixel(Pixel&& other);

        /* Allows the pixel to be indexed numerically (non-mutable). */
        HOST_DEVICE double operator[](const size_t i) const;
        /* Allows the pixel to be indexed numerically (mutable). */
        HOST_DEVICE double& operator[](const size_t i);

        /* Returns the x-position of the Pixel. */
        HOST_DEVICE inline size_t x() const { return this->pos.x; }
        /* Returns the y-position of the Pixel. */
        HOST_DEVICE inline size_t y() const { return this->pos.y; }

        /* Mark the Frame class as friend. */
        friend class Frame;
    };

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
        Frame(size_t width, size_t height, void* data);

    public:
        /* Width of the Frame (in pixels). */
        const size_t width;
        /* Height of the Frame (in pixels). */
        const size_t height;

        /* Mutable iterator for the Frame class. */
        class iterator {
        private:
            /* Index of the last-used element. */
            Coordinate pos;
            /* Width of the parent Frame. */
            size_t width;
            /* Height of the parent Frame. */
            size_t height;
            /* Reference to the internal frame. */
            Frame* data;

        public:
            /* Constructor for the iterator class, starting at the beginning (0, 0). */
            HOST_DEVICE iterator(Frame* frame);
            /* Constructor for the iterator class, starting at a specified beginning. */
            HOST_DEVICE iterator(Frame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            HOST_DEVICE inline bool operator==(const iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            HOST_DEVICE iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            HOST_DEVICE inline Pixel operator*() const { return this->data->operator[]({this->pos.x, this->pos.y}); }
        };

        /* Non-mutable iterator for the Frame class. */
        class const_iterator {
        private:
            /* Index of the last-used element. */
            Coordinate pos;
            /* Width of the parent Frame. */
            size_t width;
            /* Height of the parent Frame. */
            size_t height;
            /* Reference to the internal frame. */
            const Frame* data;

        public:
            /* Constructor for the Frame class, starting at the beginning (0, 0). */
            HOST_DEVICE const_iterator(const Frame* frame);
            /* Constructor for the Frame class, starting at a specified beginning. */
            HOST_DEVICE const_iterator(const Frame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            HOST_DEVICE inline bool operator==(const const_iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            HOST_DEVICE inline bool operator!=(const const_iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            HOST_DEVICE const_iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE const_iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE const_iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            HOST_DEVICE const_iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            HOST_DEVICE const_iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            HOST_DEVICE inline const Pixel operator*() const { return this->data->operator[]({this->pos.x, this->pos.y}); }
        };

        /* Constructor for the Frame class. Initializes an image with the specified dimensions with uninitialized pixels. */
        Frame(size_t width, size_t height);
        #ifdef CUDA
        /* CPU-side constructor for the Frame class which takes a FramePtr and uses the pointer from that as storage. */
        __device__ Frame(const FramePtr& ptr);
        #endif
        /* Copy constructor for the Frame class. */
        HOST_DEVICE Frame(const Frame& other);
        /* Move constructor for the Frame class. */
        HOST_DEVICE Frame(Frame&& other);
        /* Deconstructor for the Frame class (Host-only). */
        HOST_DEVICE ~Frame();

        /* Indexes the Frame for a given coordinate (non-mutable). */
        HOST_DEVICE const Pixel operator[](const Coordinate& index) const;
        /* Indexes the Frame for a given coordinate (mutable). */
        HOST_DEVICE Pixel operator[](const Coordinate& index);

        #ifdef CUDA
        /* Copies the Frame to the GPU. Optionally takes a point to GPU-allocated data to copy everything there. */
        FramePtr toGPU(void* data = nullptr) const;
        /* Creates a new Frame object based on the GPU data. */
        static Frame fromGPU(const FramePtr& ptr);
        #endif

        /* Writes the Frame to a PNG of our choosing. */
        void toPNG(const std::string& path) const;

        /* Copy assignment operator for the Frame class. Note that this throws a runtime_error if the Frame is not the same size. */
        HOST_DEVICE Frame& operator=(Frame other);
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

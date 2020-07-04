/* FRAME.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:24 PM
 * Last edited:
 *   7/4/2020, 2:53:12 PM
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

namespace RayTracer {
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
        Pixel(const Coordinate& pos, double* const data);
    
    public:
        /* Reference to the red-channel value. */
        double& r;
        /* Reference to the green-channel value. */
        double& g;
        /* Reference to the blue-channel value. */
        double& b;

        /* Copy constructor for the Pixel class. */
        Pixel(const Pixel& other);
        /* Move constructor for the Pixel class. */
        Pixel(Pixel&& other);

        /* Allows the pixel to be indexed numerically (non-mutable). */
        double operator[](const size_t i) const;
        /* Allows the pixel to be indexed numerically (mutable). */
        double& operator[](const size_t i);

        /* Returns the x-position of the Pixel. */
        inline size_t x() const { return this->pos.x; }
        /* Returns the y-position of the Pixel. */
        inline size_t y() const { return this->pos.y; }

        /* Mark the Frame class as friend. */
        friend class Frame;
    };

    /* A class which is used to manipulate the Pixels of an output frame. Can also write its internal buffer to a given file as PNG. */
    class Frame {
    private:
        /* The data containing all pixel values. Is height * width * 4 large. */
        double* data;

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
            iterator(Frame* frame);
            /* Constructor for the iterator class, starting at a specified beginning. */
            iterator(Frame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            inline bool operator==(const iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            inline bool operator!=(const iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            inline Pixel operator*() const { return Pixel(this->pos, this->data->data + 3 * (this->pos.y * this->data->width + this->pos.x)); }
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
            const_iterator(const Frame* frame);
            /* Constructor for the Frame class, starting at a specified beginning. */
            const_iterator(const Frame* frame, const Coordinate& pos);

            /* Checks if two iterators are equal. */
            inline bool operator==(const const_iterator& other) { return this->data == other.data && this->pos.x == other.pos.x && this->pos.y == other.pos.y; }
            /* Checks if two iterator are unequal. */
            inline bool operator!=(const const_iterator& other) { return this->data != other.data || this->pos.x != other.pos.x || this->pos.y != other.pos.y; }

            /* Increments the iterator to the next position (in the x-axis). */
            const_iterator& operator++();
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            const_iterator operator+(size_t n) const;
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            const_iterator operator+(const Coordinate& n) const;
            /* Moves the iterator n positions forward, where n is linear through the grid. */
            const_iterator& operator+=(size_t n);
            /* Moves the iterator x and y positions forward over the x- and y-axis respectively. */
            const_iterator& operator+=(const Coordinate& n);

            /* Dereferences the iterator. */
            inline const Pixel operator*() const { return Pixel(this->pos, this->data->data + 3 * (this->pos.y * this->data->width + this->pos.x)); }
        };

        /* Constructor for the Frame class. Initializes an image with the specified dimensions with uninitialized pixels. */
        Frame(size_t width, size_t height);
        /* Copy constructor for the Frame class. */
        Frame(const Frame& other);
        /* Move constructor for the Frame class. */
        Frame(Frame&& other);
        /* Deconstructor for the Frame class. */
        ~Frame();

        /* Indexes the Frame for a given coordinate (non-mutable). */
        const Pixel operator[](const Coordinate& index) const;
        /* Indexes the Frame for a given coordinate (mutable). */
        Pixel operator[](const Coordinate& index);

        /* Writes the Frame to a PNG of our choosing. */
        void toPNG(const std::string& path) const;

        /* Copy assignment operator for the Frame class. Note that this throws a runtime_error if the Frame is not the same size. */
        Frame& operator=(Frame other);
        /* Move assignment operator for the Frame class. Note that this throws a runtime_error if the Frame is not the same size. */
        Frame& operator=(Frame&& other);
        /* Swap operator for the Frame class. Note that this does not swap the size, and therefore expects the two Frames to have the same size. */
        friend void swap(Frame& f1, Frame& f2);

        /* Returns a mutable iterator to the beginning of this Frame. */
        inline iterator begin() { return Frame::iterator(this); }
        /* Returns a mutable iterator beyond the end of this Frame. */
        inline iterator end() { return Frame::iterator(this, {0, this->height}); }
        /* Returns a const_iterator to the beginning of this Frame. */
        inline const_iterator begin() const { return Frame::const_iterator(this); };
        /* Returns a const_iterator beyond the end of this Frame. */
        inline const_iterator end() const { return Frame::const_iterator(this, {0, this->height}); }
    };

    void swap(Frame& f1, Frame& f2);
}

#endif

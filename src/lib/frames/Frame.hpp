/* FRAME.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 4:47:24 PM
 * Last edited:
 *   7/1/2020, 6:00:16 PM
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
    /* A small struct allowing the user to indicate an (x, y) coordinate. */
    struct Coordinate {
        size_t x;
        size_t y;
    };

    /* A struct which wraps a Pixel, i.e., four consecutive doubles. */
    class Frame;
    struct Pixel {
    private:
        /* Pointer to the data in the Image class. */
        double* data;

        /* Constructor for the Pixel struct, only accessible from the Frame class. */
        Pixel(double* const data);
    
    public:
        /* Mark the Frame class as friend. */
        friend class Frame;

        /* Reference to the red-channel value. */
        double& r;
        /* Reference to the green-channel value. */
        double& g;
        /* Reference to the blue-channel value. */
        double& b;
        /* Reference to the alpha-channel value. */
        double& a;

        /* Copy constructor for the Pixel class. */
        Pixel(const Pixel& other);
        /* Move constructor for the Pixel class. */
        Pixel(Pixel&& other);

        /* Allows the pixel to be indexed numerically (non-mutable). */
        double operator[](const size_t i) const;
        /* Allows the pixel to be indexed numerically (mutable). */
        double& operator[](const size_t i);

        /* Copy assignment operator for the Pixel class. */
        Pixel& operator=(Pixel other);
        /* Move assignment operator for the Pixel class. */
        Pixel& operator=(Pixel&& other);
        /* Swap operator for the Pixel class. */
        friend void swap(Pixel& p1, Pixel& p2);
    };

    void swap(Pixel& p1, Pixel& p2);

    /* A class which functions as a container for the resulting image of the renderer. */
    class Frame {
    private:
        /* The data containing all pixel values. Is height * width * 4 large. */
        double* data;

    public:
        /* The width, in pixels, of this frame. */
        const size_t width;
        /* The height, in pixels, of this frame. */
        const size_t height;

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
    };

    void swap(Frame& f1, Frame& f2);
}

#endif

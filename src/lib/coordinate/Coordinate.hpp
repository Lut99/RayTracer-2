/* COORDINATE.hpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:02:10
 * Last edited:
 *   09/07/2020, 16:03:54
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Coordinate struct is meant to easily pass (x, y) coordinates to a
 *   function.
**/

#ifndef COORDINATE_HPP
#define COORDINATE_HPP

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    /* The Coordinate struct. */
    struct Coordinate {
        /* The target x-location. */
        size_t x;
        /* The target y-location. */
        size_t y;

        /* Allows the Coordinate to be swapped. */
        friend HOST_DEVICE void swap(Coordinate& c1, Coordinate& c2);
    };

    /* Allows the Coordinate to be swapped. */
    HOST_DEVICE void swap(Coordinate& c1, Coordinate& c2);
}

#endif

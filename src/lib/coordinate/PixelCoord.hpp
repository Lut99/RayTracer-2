/* PIXEL COORD.hpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:02:10
 * Last edited:
 *   13/07/2020, 13:23:34
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Coordinate struct is meant to easily pass (x, y) coordinates to a
 *   function.
**/

#ifndef PIXELCOORD_HPP
#define PIXELCOORD_HPP

#include <cstdlib>

#include "GPUDev.hpp"

namespace RayTracer {
    using pixel_coord = size_t;

    /* A struct which stores coordinates relevant for an image (integral, non-negative). */
    struct PixelCoord {
        /* The target x-location. */
        pixel_coord x;
        /* The target y-location. */
        pixel_coord y;

        /* Creates an empty PixelCoord. */
        PixelCoord();
        /* Creates a PixelCoord with given values. */
        PixelCoord(pixel_coord x, pixel_coord y);

        /* Allows the PixelCoord to be swapped. */
        friend HOST_DEVICE void swap(PixelCoord& c1, PixelCoord& c2);
    };

    /* Allows the PixelCoord to be swapped. */
    HOST_DEVICE void swap(PixelCoord& c1, PixelCoord& c2);
}

#endif

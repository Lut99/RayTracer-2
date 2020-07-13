/* PIXEL COORD.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:03:11
 * Last edited:
 *   13/07/2020, 13:23:06
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The PixelCoord struct is meant to easily pass (x, y) coordinates to a
 *   function.
**/

#include "PixelCoord.hpp"

using namespace std;
using namespace RayTracer;


/***** PIXELCOORD *****/
PixelCoord::PixelCoord() :
    x(0),
    y(0)
{}

PixelCoord::PixelCoord(pixel_coord x, pixel_coord y) :
    x(x),
    y(y)
{}



HOST_DEVICE void RayTracer::swap(PixelCoord& c1, PixelCoord& c2) {
    // Swap the x and y
    pixel_coord t = c1.x;
    c1.x = c2.x;
    c2.x = t;

    t = c1.y;
    c1.y = c2.y;
    c2.y = t;
}

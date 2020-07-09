/* COORDINATE.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 16:03:11
 * Last edited:
 *   09/07/2020, 16:04:18
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The Coordinate struct is meant to easily pass (x, y) coordinates to a
 *   function.
**/

#include "Coordinate.hpp"

using namespace std;
using namespace RayTracer;


HOST_DEVICE void RayTracer::swap(Coordinate& c1, Coordinate& c2) {
    // Swap the x and y
    double t = c1.x;
    c1.x = c2.x;
    c2.x = t;

    t = c1.y;
    c1.y = c2.y;
    c2.y = t;
}

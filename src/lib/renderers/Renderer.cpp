/* RENDERER.cpp
 *   by Lut99
 *
 * Created:
 *   16/07/2020, 17:03:51
 * Last edited:
 *   16/07/2020, 17:22:01
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Baseclass for the renderers used in this project. The idea is that this
 *   version handles as many common things as it can (especially between the
 *   sequential and CPU-optimised renderer) and allows child classes to
 *   override where needed.
**/

#include "Vec3.hpp"

#include "Renderer.hpp"

using namespace std;
using namespace RayTracer;


Renderer::Renderer()
{}


Pixel Renderer::background(const Ray& ray) {
    // Return somewhere on a vertical gradient
    Vec3 direction = ray.direction.normalize();
    double t = 0.5*(direction.y + 1.0);
    return (1.0-t) * Pixel(1.0, 1.0, 1.0) + t * Pixel(0.5, 0.7, 1.0);
}



void Renderer::update() {
    // Nothing yet
}

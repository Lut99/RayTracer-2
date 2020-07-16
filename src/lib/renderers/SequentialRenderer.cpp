/* SEQUENTIAL RENDERER.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 3:52:24 PM
 * Last edited:
 *   16/07/2020, 17:28:44
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The simplest RayTraced-renderer. Does not apply any optimisation, and
 *   is useful for testing or very simple machines.
**/

#include "Camera.hpp"

#include "SequentialRenderer.hpp"

using namespace std;
using namespace RayTracer;


SequentialRenderer::SequentialRenderer()
{}



std::vector<Frame> SequentialRenderer::render() {
    // For now, use a fixed camera
    Camera cam(Point3(0, 0, 0), Point3(1, 0, 0), Vec3(0, 1, 0), 90, 320, 200);

    // Create the to-be-returned list
    std::vector<Frame> to_return;
    for (Pixel p : to_return.at(0)) {
        p = this->background(cam.cast(p.x(), p.y()));
    }

    // Return the list
    return to_return;
}

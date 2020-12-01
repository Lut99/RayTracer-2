/* SEQUENTIAL RENDERER.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 3:52:24 PM
 * Last edited:
 *   17/07/2020, 17:20:12
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The simplest RayTraced-renderer. Does not apply any optimisation, and
 *   is useful for testing or very simple machines.
**/

#include <iostream>

#include "Camera.hpp"

#include "SequentialRenderer.hpp"

using namespace std;
using namespace RayTracer;


SequentialRenderer::SequentialRenderer()
{}



std::vector<Frame> SequentialRenderer::render() {
    cout << "Preparing... " << flush;

    // For now, use a fixed camera
    Camera cam(Point3(0, 0, 0), Point3(1, 0, 0), Vec3(0, 1, 0), 90, 320, 200);

    // Create the to-be-returned list
    std::vector<Frame> to_return;
    to_return.push_back(Frame(cam.frame_width, cam.frame_height));

    // Fill the frame with background
    cout << "Done" << endl << "Rendering frame... " << flush;
    for (Pixel p : to_return.at(0)) {
        p = this->background(cam.cast(p.x(), p.y()));
    }
    cout << "Done" << endl;

    // Return the list
    return to_return;
}



/* Creates a new renderer with general genderer options. The specific render corresponds to the backend used. */
Renderer* RayTracer::rendererFactory() {
    return (Renderer*) new SequentialRenderer();
}

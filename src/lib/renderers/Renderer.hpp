/* RENDERER.hpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 3:51:23 PM
 * Last edited:
 *   17/07/2020, 16:34:39
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Baseclass for the renderers used in this project. The idea is that this
 *   version handles as many common things as it can (especially between the
 *   sequential and CPU-optimised renderer) and allows child classes to
 *   override where needed.
**/

#ifndef RENDERER_HPP
#define RENDERER_HPP

#include <vector>

#include "Ray.hpp"
#include "Frame.hpp"

namespace RayTracer {
    /* Baseclass from which all renderers extend. */
    class Renderer {
    protected:
        /* Constructor for the Renderer. */
        Renderer();

        /* Returns the background colour based on a Ray to show a gradient. */
        virtual Pixel background(const Ray& ray);

    public:
        /* Virtual destructor for the Renderer class. */
        virtual ~Renderer() = 0;

        /* Renders a frame for each camera in the scene, and therefore returns a list of frames (as vector). */
        virtual std::vector<Frame> render() = 0;
        /* Updates the internal World object according to the programmed behaviour. */
        virtual void update();
    };

    /* Creates a new renderer with general genderer options. The specific render corresponds to the backend used. */
    Renderer* rendererFactory();
}

#endif

/* SEQUENTIAL RENDERER.hpp
 *   by Lut99
 *
 * Created:
 *   16/07/2020, 17:06:08
 * Last edited:
 *   16/07/2020, 17:36:29
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The simplest RayTraced-renderer. Does not apply any optimisation, and
 *   is useful for testing or very simple machines.
**/

#ifndef SEQUENTIALRENDERER_HPP
#define SEQUENTIALRENDERER_HPP

#include "Renderer.hpp"

namespace RayTracer {
    class SequentialRenderer : public Renderer {
    public:
        /* Constructor for a SequentialRenderer. */
        SequentialRenderer();

        /* Sequentially renders a frame for every camera in the given scene. */
        virtual std::vector<Frame> render();
    };
}

#endif

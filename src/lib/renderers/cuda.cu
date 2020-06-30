/* GPURENDERER.cu
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 3:54:38 PM
 * Last edited:
 *   6/30/2020, 4:28:09 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   A RayTraced-renderer which offloads to the GPU using Nvidia's CUDA.
 *   Can only be used on platforms with Nvidia GPUs supporting this
 *   feature.
**/

#include "Renderer.hpp"

using namespace std;
using namespace RayTracer;

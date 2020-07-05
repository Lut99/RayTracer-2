/* RAY CONTAINER.cpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:46:10
 * Last edited:
 *   05/07/2020, 17:47:05
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The RayCollection-class is used to perform bulk operations on a whole
 *   set of Rays. Most notably, it revolves around copying everything to
 *   the GPU in one go, adding Rays in a delayed fashion and matching all
 *   rays with a single object. Additionally, it also features an iterator
 *   which can be used to iterate over the internal objects.
**/

#include "RayCollection.hpp"

using namespace std;
using namespace RayTracer;




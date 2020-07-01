/* GPUTOOLS.cu
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 2:47:26 PM
 * Last edited:
 *   7/1/2020, 4:23:13 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains things that are available on the CPU, but not on
 *   the GPU. For example, this file defines some general swap operators.
**/

#include "GPUTools.hpp"

using namespace Tools;


__device__ void Tools::swap(bool& b1, bool& b2) {
    bool tb = b1;
    b1 = b2;
    b2 = tb;
}

__device__ void Tools::swap(double& f1, double& f2) {
    double tf = f1;
    f1 = f2;
    f2 = tf;
}

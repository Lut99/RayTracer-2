/* GPUTOOLS.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 2:40:31 PM
 * Last edited:
 *   7/1/2020, 4:23:04 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains things that are available on the CPU, but not on
 *   the GPU. For example, this file defines some general swap operators.
**/

#ifndef GPUTOOLS_HPP
#define GPUTOOLS_HPP

#include <type_traits>

namespace Tools {
    /***** SWAP OPERATORS *****/
    /* Swaps two pointers. */
    template <class T, typename = std::enable_if_t<std::is_pointer<T>::value> >
    __device__ void swap(T& p1, T& p2) {
        T tp = p1;
        p1 = p2;
        p2 = tp;
    }

    /* Swaps two booleans. */
    __device__ void swap(bool& b1, bool& b2);

    /* Swaps two doubles. */
    __device__ void swap(double& f1, double& f2);
}

#endif

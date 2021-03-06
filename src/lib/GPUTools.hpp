/* GPUTOOLS.hpp
 *   by Lut99
 *
 * Created:
 *   7/1/2020, 2:40:31 PM
 * Last edited:
 *   09/07/2020, 17:54:41
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

#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    /***** SWAP OPERATORS *****/
    /* Swaps two pointers. */
    template <class T, typename = std::enable_if_t<std::is_pointer<T>::value || std::is_arithmetic<T>::value> >
    HOST_DEVICE void swap(T& p1, T& p2) {
        T tp = p1;
        p1 = p2;
        p2 = tp;
    }

    // /* Swaps two arithmetic types. */
    // template <class T, typename = std::enable_if_t<std::is_arithmetic<T>::value> >
    // HOST_DEVICE void swap(T& a1, T& a2) {
    //     T ta = a1;
    //     a1 = a2;
    //     a2 = ta;
    // }
}

#endif

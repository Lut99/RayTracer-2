/* OPERATORS.cpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 17:28:51
 * Last edited:
 *   01/12/2020, 13:21:31
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file declares and defines operators used with the RayTracer's
 *   ostream class. It will be copied into the ostream.hpp file.
**/

#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <type_traits>

#include "ostream.hpp"
#include "GPUDev.hpp"

namespace RayTracer {
    /* Allows characters to be printed to the ostream. */
    HOST_DEVICE RayTracer::ostream& operator<<(RayTracer::ostream& os, const char& c);
    /* Allows character arrays to be printed to the ostream. */
    HOST_DEVICE RayTracer::ostream& operator<<(RayTracer::ostream& os, const char*& c);
    /* Allows std::strings to be printed to the ostream. */
    HOST_DEVICE RayTracer::ostream& operator<<(RayTracer::ostream& os, const std::string& s);
    /* Allows numeric types to be printed to the ostream. */
    template <class T, typename = std::enable_if_t<std::is_signed<T>::value && std::is_integral<T>::value> >
    HOST_DEVICE RayTracer::ostream& operator<<(RayTracer::ostream& os, const T& a) {
        #ifdef __CUDA_ARCH__
        /* GPU-side code. */
        
        
        #else
        /* Host-side code. */

        os.os << a;

        #endif

        return os;
    }
}

#endif

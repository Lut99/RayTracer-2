/* OSTREAM.hpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 16:23:18
 * Last edited:
 *   13/07/2020, 17:47:22
 * Auto updated?
 *   Yes
 *
 * Description:
 *   The ostream class serves as a RayTracer-only baseclass for output
 *   stream operations. While the class itself is CPU-only, it features
 *   abilities to spawn an ostream_gpu child on the GPU which collects
 *   prints and then streams them back to the CPU once ostream::sync() is
 *   called. Note that while CPU-side printing is instant, GPU-print only
 *   happens once sync() is called and therefore at the end of a kernel.
**/

#ifndef OSTREAM_HPP
#define OSTREAM_HPP

#include <ostream>
#include <type_traits>

#include "GPUDev.hpp"

namespace RayTracer {
    /* Baseclass for all CPU / GPU stream operations. */
    class ostream {
    protected:
        /* Wrapped CPU-side stream object. */
        std::ostream& os;

        #ifdef CUDA
        /* Pointer to the zero-terminated, GPU-side buffer. */
        unsigned char* buffer;
        /* Number of characters written to the GPU-side buffer. */
        size_t n_chars;
        /* Maximum size (in bytes) of the GPU-side buffer. Must be a multiple of 4. */
        size_t max_size;

        /* Writes a string of characters of given size to the internal buffer. */
        __device__ void write(size_t n, const char* str);
        /* Writes a zero-terminate string of characters to the internal buffer (including null-character). */
        __device__ void write(const char* str);
        #endif

    public:
        /* Constructor for the ostream class. Wraps any std::ostream object to write to. The maximum buffer size of the GPU can be given as option, but must be a multiple of 4 and is useless when not running on the GPU. */
        ostream(std::ostream& os, size_t max_size = 1048576);
        /* Copy constructor for the ostream class. */
        ostream(const ostream& other);
        /* Move constructor for the ostream class. */
        ostream(ostream&& other);
        /* Deconstructor for the ostream class. */
        ~ostream();

        /* Synchronizes the ostream with it's GPU-delegate. */
        virtual void sync();

        /* Copy assignment operator for the ostream class. */
        inline ostream& operator=(const ostream& other) { return *this = ostream(other); }
        /* Move assignment operator for the ostream class. */
        ostream& operator=(ostream&& other);
        /* Swap operator for the ostream class. */
        friend void swap(ostream& os1, ostream& os2);

        /* Allows characters to be printed to the ostream. */
        friend HOST_DEVICE ostream& operator<<(ostream& os, const char& c);
        /* Allows character arrays to be printed to the ostream. */
        friend HOST_DEVICE ostream& operator<<(ostream& os, const char*& c);
        /* Allows std::strings to be printed to the ostream. */
        friend HOST_DEVICE ostream& operator<<(ostream& os, const std::string& s);
        /* Allows numeric types to be printed to the ostream. */
        template <class T, typename>
        friend HOST_DEVICE ostream& operator<<(ostream& os, const T& a);
    };

    /* Swap operator for the ostream class. */
    void swap(ostream& os1, ostream& os2);
}

#endif

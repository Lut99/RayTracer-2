/* RAY COLLECTION.hpp
 *   by Lut99
 *
 * Created:
 *   05/07/2020, 17:46:28
 * Last edited:
 *   05/07/2020, 17:54:24
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

#ifndef RAYCOLLECTION_HPP
#define RAYCOLLECTION_HPP

#include "Ray.hpp"


#ifdef CUDA
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

namespace RayTracer {
    /* Struct which contains all the data necessary to move the RayCollection between the host and device. */
    struct RayCollectionPtr {
        void* data;
        void* queue;
        size_t n_rays;  
    };

    class RayCollection {
    private:
        /* Reference to the data contained by this Collection. */
        Ray* data;
        /* Reference to the list used to queue new Rays. */
        Ray* queue;
        /* Determines if the Collection outsources memory management to something external or does it by itself. */
        bool is_external;
        
    public:
        /* Constructor for the RayCollection class, which takes the number of rays cast per frame. */
        HOST_DEVICE RayCollection(size_t n_rays);
        #ifdef CUDA
        /* Constructor for the RayCollection class, which takes a RayCollection pointer which allows it to oursource memory management. */
        __device__ RayCollection(const RayCollectionPtr& ptr);
        #endif
        /* Copy constructor for the RayCollection class. */
        HOST_DEVICE RayCollection(const RayCollection& other);
        /* Move constructor for the RayCollection class. */
        HOST_DEVICE RayCollection(RayCollection&& other);

        // TODO: The rest
    };
}

#endif

/* TEST RAY.cpp
 *   by Lut99
 *
 * Created:
 *   07/07/2020, 17:15:03
 * Last edited:
 *   07/07/2020, 17:35:33
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Provides tests for the Ray class, and in particular the GPU-related
 *   memory management.
**/

#include <iostream>

#include "Ray.hpp"

using namespace std;
using namespace RayTracer;

#define TOSTR(S) _TOSTR(S)
#define _TOSTR(S) #S
#define ASSERT(EXPR) \
    if (!(EXPR)) { \
        cout << "[FAIL]" << endl << endl; \
        cerr << "ERROR: Expression '" TOSTR(EXPR) "' is incorrect." << endl << endl; \
        return false; \
    }
#ifdef CUDA
#define CUDA_ASSERT(ID) \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        cout << "[FAIL]" << endl << endl; \
        cerr << "ERROR: " TOSTR(ID) ": " << cudaGetErrorString(cudaGetLastError()) << endl << endl; \
        return false; \
    }
#endif


#ifdef CUDA
__global__ void test_gpu_kernel(Point3& result, Ray* test1_ptr, Ray* test2_ptr, Ray* test3_ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        // Only get the result of at() from test3_ptr
        Ray& test3 = *test3_ptr;

        result = test3.at(42);
    }
}

bool test_gpu() {
    cout << "   Testing CPU / GPU portability...    " << flush;

    // Create a default GPU-Ray
    Ray* test1 = Ray::GPU_create();
    // Create a default ray with custom vectors
    Ray* test2 = Ray::GPU_create(Point3(1, 2, 3), Vec3(1, 2, 3));
    // Create a ray which is a copy of a CPU-side ray
    Ray test3_cpu(Point3(4, 5, 6), Vec3(4, 5, 6));
    Ray* test3 = Ray::GPU_create(test3_cpu);

    // Run the kernel
    Point3 result;
    test_gpu_kernel<<<1, 32>>>(result, test1, test2, test3);
    cudaDeviceSynchronize();
    CUDA_ASSERT(test_gpu_kernel);

    // Copy the results back & free 'em
    Ray result1 = Ray::GPU_copy(test1);
    Ray result2 = Ray::GPU_copy(test2);
    Ray result3 = Ray::GPU_copy(test3);
    Ray::GPU_free(test1);
    Ray::GPU_free(test2);
    Ray::GPU_free(test3);

    // Compare them
    ASSERT(Ray() == result1);
    ASSERT(Ray(Point3(1, 2, 3), Vec3(1, 2, 3)) == result2);
    ASSERT(test3_cpu == result3);

    // Also the result
    ASSERT(test3_cpu.at(42) == result);

    // Done, return
    cout << "[ OK ]" << endl;
    return true;
}
#endif


int main() {
    #ifdef CUDA
    test_gpu();
    #endif
    
    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

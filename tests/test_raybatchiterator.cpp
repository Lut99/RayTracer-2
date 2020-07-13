/* TEST RAYBATCHITERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 17:59:37
 * Last edited:
 *   13/07/2020, 15:27:31
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This class is used to test the iteration correctness of the
 *   RayBatchIterator, both on the CPU and the GPU.
**/

#include <iostream>

#include "Vec3.hpp"
#include "Camera.hpp"
#include "RayBatchIterator.hpp"

#include "Frame.hpp"

using namespace std;
using namespace RayTracer;

#define TOSTR(S) _TOSTR(S)
#define _TOSTR(S) #S
#ifdef CUDA
#define CUDA_TEST_ASSERT(ID) \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        cout << "[FAIL]" << endl << endl; \
        cerr << "ERROR: " TOSTR(ID) ": " << cudaGetErrorString(cudaGetLastError()) << endl << endl; \
        return false; \
    }
#endif


bool test_iteration() {
    cout << "   Testing iteration correctness...    " << flush;

    // Create an example camera
    Camera cam(Point3(1, 2, 3), Point3(4, 5, 6), Vec3(0, 1, 0), 90, 500, 250);

    // Iterate over the possible rays, compare with the directly casted rays.
    RayBatchIterator iter(cam, 1);
    Point2 pos;
    size_t batch_i = 0;
    for (RayBatch batch : iter) {
        for (Ray& ray : batch) {
            Ray gold = cam.cast(pos.x, pos.y, false);
            if (ray != gold) {
                cout << "[FAIL]" << endl << endl;
                cerr << "ERROR: " << ray << " is not equal to " << gold << " @ (" << pos.x << ", " << pos.y << ", batch " << batch_i << ")" << endl << endl;
                return false;
            }

            pos++;
            pos = pos.rebalance(500);
        }
        batch_i++;
    }

    cout << "[ OK ]" << endl;
    return true;
}


#ifdef CUDA
__global__ void test_gpu_kernel(bool* success_ptr, Camera* cam_ptr, RayBatch* batch_ptr, size_t batch_i) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i == 0) {
        // Get references
        bool& success = *success_ptr;
        Camera& cam = *cam_ptr;
        RayBatch& batch = *batch_ptr;
        success = true;

        // Loop through and compare the x, y with the expected
        Point2 pos = Point2(batch_i * 500000, 0).rebalance(500);
        size_t r = 0;
        for (Ray& ray : batch) {
            Ray gold = cam.cast(pos, false);
            if (ray != gold) {
                printf("[FAIL]\n\n");
                printf("ERROR: GPU: Iterator does not return correct ray @ (%lu, %lu, batch = %lu): got [Ray : orig [%f, %f, %f], dir [%f, %f, %f]], expected [Ray : orig [%f, %f, %f], dir [%f, %f, %f]]\n\n",
                       pos.x, pos.y, batch_i,
                       ray.origin.x, ray.origin.y, ray.origin.z, ray.direction.x, ray.direction.y, ray.direction.z,
                       gold.origin.x, gold.origin.y, gold.origin.z, gold.direction.x, gold.direction.y, gold.direction.z);
                success = false;
                return;
            }

            r++;
            if (r == 5) {
                r = 0;
                pos++;
                pos = pos.rebalance(500);
            }
        }
    }
}


bool test_gpu() {
    cout << "   Testing CPU / GPU portability...    " << flush;
    
    // Create an example camera
    Camera cam(Point3(1, 2, 3), Point3(4, 5, 6), Vec3(0, 1, 0), 90, 500, 250);

    // Create the iterator
    RayBatchIterator iter(cam, 5);

    // Iterator over it, but then using the gpu at each step
    size_t i = 0;
    for (RayBatch batch : iter) {
        // Copy stuff to the GPU
        Camera* cam_gpu = Camera::GPU_create(cam);
        RayBatch* gpu = RayBatch::GPU_create(batch);
        bool* success_ptr;
        cudaMalloc((void**) &success_ptr, sizeof(bool));

        // Run the kernel
        test_gpu_kernel<<<1, 32>>>(success_ptr, cam_gpu, gpu, i);
        cudaDeviceSynchronize();
        CUDA_TEST_ASSERT(test_gpu_kernel);

        // Free the camera
        Camera::GPU_free(cam_gpu);

        // Copy & free the GPU-side RayBatch
        RayBatch result = RayBatch::GPU_copy(gpu);
        RayBatch::GPU_free(gpu);
        
        // Copy the success back to check if success
        bool success;
        cudaMemcpy((void*) &success, (void*) success_ptr, sizeof(bool), cudaMemcpyDeviceToHost);
        cudaFree(success_ptr);
        if (!success) { return false; }

        // Check if this result is the same
        RayBatch::iterator result_i = result.begin();
        RayBatch::iterator batch_i = batch.begin();
        size_t x = 0;
        size_t y = 0;
        for (; result_i != result.end() && batch_i != batch.end(); ) {
            // Check if the same
            if (*result_i != *batch_i) {
                cout << "[FAIL]" << endl << endl;
                cerr << "ERROR: CPU after copying: elements are not the same for x = " << x << ", y = " << y << " (batch " << i << ")" << endl << endl;
                return false;
            }

            // Advance
            ++result_i;
            ++batch_i;
            y += (x + 1) / 500;
            x = (x + 1) % 500;
        }
        i++;
    }

    cout << "[ OK ]" << endl;
    return true;
}
#endif


int main() {
    // test_iteration();
    #ifdef CUDA
    test_gpu();
    #endif

    cout << "Done." << endl << endl;
    return 0;
}

/* TEST RAYBATCHITERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 17:59:37
 * Last edited:
 *   09/07/2020, 18:01:01
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
#define CUDA_ASSERT(ID) \
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
    size_t x = 0;
    size_t y = 0;
    size_t batch_i = 0;
    for (RayBatch batch : iter) {
        for (Ray& ray : batch) {
            Ray gold = cam.cast(x, y);
            if (ray != gold) {
                cout << "[FAIL]" << endl << endl;
                cerr << "ERROR: " << ray << " is not equal to " << gold << " @ (" << x << ", " << y << ", batch " << batch_i << ")" << endl << endl;
                return false;
            }

            y += (x + 1) / 500;
            x = (x + 1) % 500;
        }
        batch_i++;
    }

    cout << "[ OK ]" << endl;
    return true;
}


int main() {
    test_iteration();

    cout << "Done." << endl << endl;
    return 0;
}

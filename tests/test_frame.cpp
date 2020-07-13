/* TEST FRAME.cpp
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 12:08:25 PM
 * Last edited:
 *   13/07/2020, 12:44:38
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This class tests the frame class. However, it does not do so
 *   automated, but simply generates a test frame, writes it to PNG, and
 *   then relies on the user to verify its correctness.
**/

#include "Frame.hpp"

#include <iostream>

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

bool test_values() {
    cout << "   Testing values of Frame class...    " << flush;

    // Create the values
    double values[50][100][3];
    for (int y = 0; y < 50; y++) {
        for (int x = 0; x < 100; x++) {
            for (int z = 0; z < 3; z++) {
                values[y][x][z] = x / 100;
            }
        }
    }

    // Create the Frame
    Frame frame(100, 50);
    for (Pixel p : frame) {
        for (size_t z = 0; z < 3; z++) {
            p[z] = values[p.y()][p.x()][z];
        }
    }

    // Compare the values
    bool succes = true;
    size_t fx, fy, fz;
    for (Pixel p : frame) {
        for (size_t z = 0; z < 3; z++) {
            if (values[p.y()][p.x()][z] != p[z]) {
                succes = false;
                fx = p.x();
                fy = p.y();
                fz = z;
                break;
            }
        }
        if (!succes) {break;}
    }

    if (succes) {
        cout << "[ OK ]" << endl;
        return true;
    } else {
        cout << "[FAIL]" << endl << endl;
        cerr << "ERROR: Expected " << values[fy][fx][fz] << " @ (" << fx << "," << fy << "," << fz << "), got " << frame[{fx, fy}][fz] << "." << endl << endl;
        return false;
    }
}

bool test_png() {
    cout << "   Writing test Frame...               " << flush;

    size_t w, h;
    w = 200;
    h = 100;

    Frame frame(w, h);

    // Write a gradient to the frame
    for (Pixel p : frame) {
        p.r = float(p.x()) / float(w);
        p.g = float((h - 1) - p.y()) / float(h);
        p.b = 0.25;
    }

    // Write the image
    frame.toPNG("tests/test.png");

    cout << "[ OK ]" << endl;
    return true;
}

#ifdef CUDA
__global__ void test_kernel(Frame* test1_ptr, Frame* test2_ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        // Create references
        Frame& test1 = *test1_ptr;
        Frame& test2 = *test2_ptr;

        // First, write a gradient to test1
        for (Pixel p : test1) {
            p.r = float(p.x()) / float(test1.width);
            p.g = float((test1.height - 1) - p.y()) / float(test1.height);
            p.b = 0.25;
        }
        // Also, darken test2
        for (Pixel p : test2) {
            p.r *= 0.5;
            p.g *= 0.5;
            p.b *= 0.5;
        }

        // Done
    }
}

bool test_values_gpu() {
    cout << "   Testing CPU / GPU portability...    " << flush;

    size_t width = 1920;
    size_t height = 1200;

    // Create empty CPU & GPU-side frames
    Frame cpu_test1(width, height);
    Frame* gpu_test1 = Frame::GPU_create(width, height);

    // Now create a CPU-side test2 frame with already a gradient
    Frame cpu_test2(width, height);
    for (Pixel p : cpu_test2) {
        p.r = float(p.x()) / float(width);
        p.g = float((height - 1) - p.y()) / float(height);
        p.b = 0.25;
    }

    // Create a GPU-counterpart
    Frame* gpu_test2 = Frame::GPU_create(cpu_test2);

    // Run the test kernel
    test_kernel<<<1, 32>>>(gpu_test1, gpu_test2);

    // Meanwhile, do the same operations here on the CPU
    // First, write a gradient to test1
    for (Pixel p : cpu_test1) {
        p.r = float(p.x()) / float(width);
        p.g = float((height - 1) - p.y()) / float(height);
        p.b = 0.25;
    }
    // Also, darken test2
    for (Pixel p : cpu_test2) {
        p.r *= 0.5;
        p.g *= 0.5;
        p.b *= 0.5;
    }

    // Synchronize with the device, check for errors
    cudaDeviceSynchronize();
    CUDA_TEST_ASSERT(test_kernel);

    // Copy the frames back from the GPU and free them over there
    Frame gpu_result1 = Frame::GPU_copy(gpu_test1);
    Frame gpu_result2 = Frame::GPU_copy(gpu_test2);
    Frame::GPU_free(gpu_test1);
    Frame::GPU_free(gpu_test2);

    // Compare them with their CPU counterparts
    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            Pixel cpu1 = cpu_test1[{x, y}];
            Pixel gpu1 = gpu_result1[{x, y}];
            if (cpu1 != gpu1) {
                cout << "[FAIL]" << endl << endl;
                cerr << "ERROR: Pixel mismatch for test1 @ (" << x << ", " << y << "): expected " << cpu1 << ", got " << gpu1 << "." << endl << endl;
                return false;
            }

            Pixel cpu2 = cpu_test2[{x, y}];
            Pixel gpu2 = gpu_result2[{x, y}];
            if (cpu2 != gpu2) {
                cout << "[FAIL]" << endl << endl;
                cerr << "ERROR: Pixel mismatch for test2 @ (" << x << ", " << y << "): expected " << cpu2 << ", got " << gpu2 << "." << endl << endl;
                return false;
            }
        }
    }

    // Succes!
    cout << "[ OK ]" << endl;
    return true;
}
#endif


int main() {
    test_values();
    test_png();
    #ifdef CUDA
    test_values_gpu();
    #endif

    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

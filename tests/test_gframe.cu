/* TEST GFRAME.cu
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 3:53:09 PM
 * Last edited:
 *   7/4/2020, 4:19:30 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Test the GPU-side Frame, together with the interaction between the CPU
 *   and GPU-side.
**/

#include <iostream>

#include "Frame.hpp"
#include "GFrame.hpp"

using namespace std;
using namespace RayTracer;


__global__ void test_values_kernel(double* values_ptr, size_t values_pitch, cudaPitchedPtr ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        GFrame frame(ptr);

        // Compare the values
        bool succes = true;
        size_t fx, fy, fz;
        for (GPixel p : frame) {
            for (size_t z = 0; z < 3; z++) {
                if (values_ptr[p.y() * values_pitch + p.x() * 3 + z] != p[z]) {
                    succes = false;
                    fx = p.x();
                    fy = p.y();
                    fz = z;
                    break;
                }
            }
            if (!succes) {break;}
        }

        if (!succes) {
            printf("[FAIL]\n\n");
            printf("ERROR: GPU: Expected %f @ (%lu, %lu, %lu), got %f.\n\n", values_ptr[fy * values_pitch + fx * 3 + fz], fx, fy, fz, frame[{fx, fy}][fz]);
        }
    }
}

bool test_values() {
    cout << "   Testing values of GFrame class...   " << flush;

    // Create the values
    double values[50][100][3];
    for (int y = 0; y < 50; y++) {
        for (int x = 0; x < 100; x++) {
            for (int z = 0; z < 3; z++) {
                values[y][x][z] = x / 100;
            }
        }
    }
    // Copy the values to the GPU
    double* values_ptr;
    size_t values_pitch;
    cudaMallocPitch((void**) &values_ptr, &values_pitch, sizeof(double) * 300, 50);
    cudaMemcpy2D((void*) values_ptr, values_pitch, values, sizeof(double) * 300, sizeof(double) * 300, 50, cudaMemcpyHostToDevice);

    // Create the Frame
    Frame frame(100, 50);
    for (Pixel p : frame) {
        for (size_t z = 0; z < 3; z++) {
            p[z] = values[p.y()][p.x()][z];
        }
    }
    // Also copy to GPU
    cudaPitchedPtr frame_ptr = GFrame::toGPU(frame);

    // Call the kernel, let it do the printing
    test_values_kernel<<<1, 32>>>(values_ptr, values_pitch, frame_ptr);

    // Retrieve the frame from the GPU
    Frame result = GFrame::fromGPU(frame_ptr);

    // Clean the values memory
    cudaFree(values_ptr);

    // Do a local comparison tp verify the copying went right
    bool succes = true;
    size_t fx, fy, fz;
    for (Pixel p : result) {
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
        cerr << "ERROR: CPU: Expected " << values[fy][fx][fz] << " @ (" << fx << "," << fy << "," << fz << "), got " << result[{fx, fy}][fz] << "." << endl << endl;
        return false;
    }
}


int main() {
    test_values();
    cudaDeviceSynchronize();

    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

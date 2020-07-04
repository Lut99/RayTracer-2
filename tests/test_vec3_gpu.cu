/* TEST VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:40:27 PM
 * Last edited:
 *   7/4/2020, 5:58:58 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Contains tests for the Vec3 class, only then the GPU-side.
**/

#include <iostream>

#include "Vec3.hpp"

using namespace std;
using namespace RayTracer;


#define TOSTR(S) _TOSTR(S)
#define _TOSTR(S) #S
#define ASSERT(EXPR) \
    if (!(EXPR)) { \
        printf("[FAIL]\n\n"); \
        printf("ERROR: Expression '" TOSTR(EXPR) "' is incorrect.\n\n"); \
        return; \
    } 


__global__ void test_equal() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing equality functions...       ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);
        Vec3 test2(1, 2, 3);
        Vec3 test3(4, 5, 6);

        // Try some
        ASSERT(test1 == test2)
        ASSERT(!(test1 == test3))
        ASSERT(!(test1 != test2))
        ASSERT(test1 != test3)
        
        ASSERT(test1 < 4)
        ASSERT(!(test1 >= 4))
        ASSERT(!(test1 < 1))
        ASSERT(test1 >= 1)

        printf("[ OK ]\n");
    }
}

__global__ void test_sum() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing addition...                 ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);
        Vec3 test2(4, 5, 6);

        // Try some
        ASSERT(test1 + 5 == Vec3(6, 7, 8))
        test1 += 5;
        ASSERT(test1 == Vec3(6, 7, 8))
        
        ASSERT(test1 + test2 == Vec3(10, 12, 14))
        test1 += test2;
        ASSERT(test1 == Vec3(10, 12, 14))

        printf("[ OK ]\n");
    }
}

__global__ void test_sub() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing subtraction...              ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);
        Vec3 test2(4, 5, 6);

        // Try some
        ASSERT(test1 - 5 == Vec3(-4, -3, -2))
        test1 -= 5;
        ASSERT(test1 == Vec3(-4, -3, -2))
        
        ASSERT(test1 - test2 == Vec3(-8, -8, -8))
        test1 -= test2;
        ASSERT(test1 == Vec3(-8, -8, -8))
        ASSERT(-test1 == Vec3(8, 8, 8))

        printf("[ OK ]\n");
    }
}

__global__ void test_mul() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing multiplication...           ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);
        Vec3 test2(4, 5, 6);

        // Try some
        ASSERT(test1 * 5 == Vec3(5, 10, 15))
        test1 *= 5;
        ASSERT(test1 == Vec3(5, 10, 15))
        
        ASSERT(test1 * test2 == Vec3(20, 50, 90))
        test1 *= test2;
        ASSERT(test1 == Vec3(20, 50, 90))

        printf("[ OK ]\n");
    }
}

__global__ void test_div() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing dividation...               ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);
        Vec3 test2(4, 5, 6);

        // Try some
        ASSERT(test1 / 5 == Vec3(1.0 / 5, 2.0 / 5, 3.0 / 5))
        test1 /= 5;
        ASSERT(fabs(test1 - Vec3(1.0 / 5, 2.0 / 5, 3.0 / 5)) <= 0.000000000000001)
        
        ASSERT(fabs(test1 / test2 - Vec3(1.0 / 5 / 4, 2.0 / 5 / 5, 3.0 / 5 / 6)) <= 0.0000000000000001)
        test1 /= test2;
        ASSERT(fabs(test1 - Vec3(1.0 / 5 / 4, 2.0 / 5 / 5, 3.0 / 5 / 6)) < 0.00000001)

        printf("[ OK ]\n");
    }
}

__global__ void test_misc() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing miscellaneous operations... ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);

        // Try some
        ASSERT(test1.sum() == 6)
        ASSERT(fabs(test1.length() - 3.74165738677) < 0.000001)
        ASSERT(test1.length_pow2() == 14)

        printf("[ OK ]\n");
    }
}

__global__ void test_access() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing access operations...        ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);

        // Try some
        ASSERT(test1[0] == test1.x)
        ASSERT(test1[1] == test1.y)
        ASSERT(test1[2] == test1.z)
        test1[0] = 10;
        test1[1] = 15;
        test1[2] = 20;
        ASSERT(test1[0] == test1.x)
        ASSERT(test1[1] == test1.y)
        ASSERT(test1[2] == test1.z)

        printf("[ OK ]\n");
    }
}

__global__ void test_math() {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        printf("   Testing math operations...          ");

        // Create a couply to try
        Vec3 test1(1, 2, 3);

        // Try some
        ASSERT(exp(test1) == Vec3(exp(1.0), exp(2.0), exp(3.0)))
        ASSERT(sqrt(test1) == Vec3(sqrt(1.0), sqrt(2.0), sqrt(3.0)))
        ASSERT(pow(test1, 4) == Vec3(pow(1.0, 4.0), pow(2.0, 4.0), pow(3.0, 4.0)))
        ASSERT(fabs(-test1) == Vec3(fabs(-1.0), fabs(-2.0), fabs(-3.0)))

        printf("[ OK ]\n");
    }
}

__global__ void test_copy_kernel(void* ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i == 0) {
        // Initialize with a ptr
        Vec3 test1(ptr);
        
        // Do some
        test1 += 5;
        test1 -= 20;
        test1 += Vec3(5, 10, 50);
    }
}

void test_copy() {
    printf("   Testing Vec3 CPU / GPU portability... ");

    // Create a CPU-side Vector
    Vec3 test1(1, 2, 3);
    
    void* ptr = test1.toGPU();

    // Run the kernel
    test_copy_kernel<<<1, 32>>>(ptr);

    // Create a new vector based on the GPU data
    Vec3 result = Vec3::fromGPU(ptr);

    // Check if it is expected
    ASSERT(result == Vec3(1, 2, 3) + 5 - 20 + Vec3(5, 10, 50))

    printf("[ OK ]\n");
}


int main() {
    test_equal<<<1, 32>>>();
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }

    test_sum<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_sub<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_mul<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_div<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_misc<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_access<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_math<<<1, 32>>>();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }
    
    test_copy();
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) { cerr << "[FAIL]" << endl << endl << "ERROR: " << cudaGetErrorString(err) << endl << endl; return EXIT_FAILURE; }

    printf("Done.\n\n");
    return EXIT_SUCCESS;
}

/* TEST VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:40:27 PM
 * Last edited:
 *   07/07/2020, 14:11:13
 * Auto updated?
 *   Yes
 *
 * Description:
 *   Contains tests for the Vec3 class.
**/

#include <iostream>

#include "Vec3.hpp"

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


bool test_equal() {
    cout << "   Testing equality functions...       " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_sum() {
    cout << "   Testing addition...                 " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_sub() {
    cout << "   Testing subtraction...              " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_mul() {
    cout << "   Testing multiplication...           " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_div() {
    cout << "   Testing dividation...               " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_misc() {
    cout << "   Testing miscellaneous operations... " << flush;

    // Create a couply to try
    Vec3 test1(1, 2, 3);

    // Try some
    ASSERT(test1.sum() == 6)
    ASSERT(fabs(test1.length() - 3.74165738677) < 0.000001)
    ASSERT(test1.length_pow2() == 14)

    cout << "[ OK ]" << endl;
    return true;
}

bool test_access() {
    cout << "   Testing access operations...        " << flush;

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

    cout << "[ OK ]" << endl;
    return true;
}

bool test_math() {
    cout << "   Testing math operations...          " << flush;

    // Create a couply to try
    Vec3 test1(1, 2, 3);

    // Try some
    ASSERT(exp(test1) == Vec3(exp(1), exp(2), exp(3)))
    ASSERT(sqrt(test1) == Vec3(sqrt(1), sqrt(2), sqrt(3)))
    ASSERT(pow(test1, 4) == Vec3(pow(1, 4), pow(2, 4), pow(3, 4)))
    ASSERT(fabs(-test1) == Vec3(fabs(-1), fabs(-2), fabs(-3)))

    cout << "[ OK ]" << endl;
    return true;
}

#ifdef CUDA
__global__ void test_copy_kernel(Vec3* test1_ptr, Vec3* test2_ptr, Vec3* test3_ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i == 0) {
        // Get references to the given objects
        Vec3& test1 = *test1_ptr;
        Vec3& test2 = *test2_ptr;
        Vec3& test3 = *test3_ptr;

        // // Do the tests
        test1 += 5;
        test1 -= 20;
        test1 += Vec3(5, 10, 50);
        test2 += 5;
        test2 -= 20;
        test2 += Vec3(5, 10, 50);
        test3 += 5;
        test3 -= 20;
        test3 += Vec3(5, 10, 50);

        // Done
    }
}

bool test_copy() {
    cout << "   Testing CPU / GPU portability...    " << flush;

    // Create an empty GPU vector.
    Vec3* test1 = Vec3::GPU_create();

    // Create a GPU-side vector with data
    Vec3* test2 = Vec3::GPU_create(1, 2, 3);

    // Create a CPU-side Vector and copy that to the GPU
    Vec3 cpu_test3(4, 5, 6);
    Vec3* test3 = Vec3::GPU_create(cpu_test3);

    // Run the kernel
    test_copy_kernel<<<1, 32>>>(test1, test2, test3);
    cudaDeviceSynchronize();
    CUDA_ASSERT(test_copy_kernel);

    // Copy all vectors back to the CPU
    Vec3 result1 = Vec3::GPU_copy(test1);
    Vec3 result2 = Vec3::GPU_copy(test2);
    Vec3 result3 = Vec3::GPU_copy(test3);
    
    // Clean the GPU-side ones
    Vec3::GPU_destroy(test1);
    Vec3::GPU_destroy(test2);
    Vec3::GPU_destroy(test3);

    // Check if it is expected
    ASSERT(result1 == Vec3() + 5 - 20 + Vec3(5, 10, 50));
    ASSERT(result2 == Vec3(1, 2, 3) + 5 - 20 + Vec3(5, 10, 50));
    ASSERT(result3 == Vec3(4, 5, 6) + 5 - 20 + Vec3(5, 10, 50));

    cout << "[ OK ]" << endl;
    return true;
}
#endif


int main() {
    test_equal();
    test_sum();
    test_sub();
    test_mul();
    test_div();
    test_misc();
    test_access();
    test_math();
    #ifdef CUDA
    test_copy();
    #endif

    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

/* TEST POINT 2.cpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 14:15:26
 * Last edited:
 *   13/07/2020, 16:41:30
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file tests the functionality of the Point2 class.
**/

#include <iostream>

#include "Point2.hpp"

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
#define CUDA_TEST_ASSERT(ID) \
    if (cudaPeekAtLastError() != cudaSuccess) { \
        cout << "[FAIL]" << endl << endl; \
        cerr << "ERROR: " TOSTR(ID) ": " << cudaGetErrorString(cudaGetLastError()) << endl << endl; \
        return false; \
    }
#endif


bool test_equal() {
    // Create a couply to try
    Point2 test1(1, 2);
    Point2 test2(1, 2);
    Point2 test3(3, 4);
    Point2 test4(2, 2);

    // Try some
    ASSERT(test1 == test2);
    ASSERT(!(test1 == test3));
    ASSERT(!(test1 != test2));
    ASSERT(test1 != test3);
    
    ASSERT(test1 < test4);
    ASSERT(!(test1 > test4));
    ASSERT(test1 <= test4);
    ASSERT(!(test1 >= test4));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_sum() {
    // Create a couply to try
    Point2 test1(1, 2);
    Point2 test2(3, 4);

    // Try some
    ASSERT(test1 + test2 == Point2(4, 6));
    test1 += test2;
    ASSERT(test1 == Point2(4, 6));
    ASSERT(++test1 == Point2(5, 6));
    ASSERT(test1++ == Point2(5, 6));
    ASSERT(test1 == Point2(6, 6));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_sub() {
    // Create a couply to try
    Point2 test1(1, 2);
    Point2 test2(3, 4);

    // Try some
    ASSERT(test2 - test1 == Point2(2, 2));
    test2 -= test1;
    ASSERT(test2 == Point2(2, 2));
    ASSERT(--test2 == Point2(1, 2));
    ASSERT(test2-- == Point2(1, 2));
    ASSERT(test2 == Point2(0, 2));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_mul() {
    // Create a couply to try
    Point2 test1(1, 2);
    Point2 test2(3, 4);

    // Try some
    ASSERT(test1 * test2 == Point2(3, 8));
    test1 *= test2;
    ASSERT(test1 == Point2(3, 8));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_div() {
    // Create a couply to try
    Point2 test1(1, 2);
    Point2 test2(3, 4);

    // Try some
    ASSERT(test2 / test1 == Point2(3, 2));
    test2 /= test1;
    ASSERT(test2 == Point2(3, 2));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_misc() {
    // Create a couply to try
    Point2 test1(1000, 0);

    // Try some
    ASSERT(test1.balance(50) == Point2(0, 20));
    test1.rebalance(50);
    ASSERT(test1 == Point2(0, 20));

    cout << "[ OK ]" << endl;
    return true;
}

bool test_access() {
    // Create a couply to try
    Point2 test1(1, 2);

    // Try some
    ASSERT(test1[0] == test1.x)
    ASSERT(test1[1] == test1.y)
    test1[0] = 10;
    test1[1] = 15;
    ASSERT(test1[0] == test1.x)
    ASSERT(test1[1] == test1.y)

    cout << "[ OK ]" << endl;
    return true;
}

#ifdef CUDA
__global__ void test_copy_kernel(Point2* test1_ptr, Point2* test2_ptr, Point2* test3_ptr) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i == 0) {
        // Get references to the given objects
        Point2& test1 = *test1_ptr;
        Point2& test2 = *test2_ptr;
        Point2& test3 = *test3_ptr;

        // // Do the tests
        test1 += Point2(5, 10);
        test2 += Point2(5, 10);
        test3 += Point2(5, 10);

        // Done
    }
}

bool test_copy() {
    // Create an empty GPU vector.
    Point2* test1 = Point2::GPU_create();

    // Create a GPU-side vector with data
    Point2* test2 = Point2::GPU_create(1, 2);

    // Create a CPU-side Vector and copy that to the GPU
    Point2 cpu_test3(3, 4);
    Point2* test3 = Point2::GPU_create(cpu_test3);

    // Run the kernel
    test_copy_kernel<<<1, 32>>>(test1, test2, test3);
    cudaDeviceSynchronize();
    CUDA_TEST_ASSERT(test_copy_kernel);

    // Copy all vectors back to the CPU
    Point2 result1 = Point2::GPU_copy(test1);
    Point2 result2 = Point2::GPU_copy(test2);
    Point2 result3 = Point2::GPU_copy(test3);
    
    // Clean the GPU-side ones
    Point2::GPU_free(test1);
    Point2::GPU_free(test2);
    Point2::GPU_free(test3);

    // Check if it is expected
    ASSERT(result1 == Point2() + Point2(5, 10));
    ASSERT(result2 == Point2(1, 2) + Point2(5, 10));
    ASSERT(result3 == Point2(3, 4) + Point2(5, 10));

    cout << "[ OK ]" << endl;
    return true;
}
#endif


int main() {
    cout << "   Testing equality functions...              " << flush;
    test_equal();

    cout << "   Testing addition...                        " << flush;
    test_sum();

    cout << "   Testing subtraction...                     " << flush;
    test_sub();

    cout << "   Testing multiplication...                  " << flush;
    test_mul();

    cout << "   Testing dividation...                      " << flush;
    test_div();

    cout << "   Testing miscellaneous operations...        " << flush;
    test_misc();

    cout << "   Testing access operations...               " << flush;
    test_access();
    
    cout << "   Testing CPU / GPU portability...           " << flush;
    #ifdef CUDA
    test_copy();
    #else
    cout << "[SKIP]" << endl;
    #endif

    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

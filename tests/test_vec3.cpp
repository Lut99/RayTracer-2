/* TEST VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:40:27 PM
 * Last edited:
 *   7/1/2020, 12:50:02 PM
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
    ASSERT(test1 == Vec3(1.0 / 5, 2.0 / 5, 3.0 / 5))
    
    ASSERT(test1 / test2 == Vec3(1.0 / 5 / 4, 2.0 / 5 / 5, 3.0 / 5 / 6))
    test1 /= test2;
    ASSERT(fabsf(test1 - Vec3(1.0 / 5 / 4, 2.0 / 5 / 5, 3.0 / 5 / 6)) < 0.00000001)

    cout << "[ OK ]" << endl;
    return true;
}

bool test_misc() {
    cout << "   Testing miscellaneous operations... " << flush;

    // Create a couply to try
    Vec3 test1(1, 2, 3);

    // Try some
    ASSERT(test1.sum() == 6)
    ASSERT(fabsf(test1.length() - 3.74165738677) < 0.000001)
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
    ASSERT(expf(test1) == Vec3(expf(1), expf(2), expf(3)))
    ASSERT(sqrtf(test1) == Vec3(sqrtf(1), sqrtf(2), sqrtf(3)))
    ASSERT(powf(test1, 4) == Vec3(powf(1, 4), powf(2, 4), powf(3, 4)))
    ASSERT(fabsf(-test1) == Vec3(fabsf(-1), fabsf(-2), fabsf(-3)))

    cout << "[ OK ]" << endl;
    return true;
}


int main() {
    test_equal();
    test_sum();
    test_sub();
    test_mul();
    test_div();
    test_misc();
    test_access();
    test_math();

    cout << "Done." << endl;
}

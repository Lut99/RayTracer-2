/* TEST VEC 3.cpp
 *   by Lut99
 *
 * Created:
 *   6/30/2020, 5:40:27 PM
 * Last edited:
 *   6/30/2020, 6:11:16 PM
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
    cout << "   Testing equality functions... " << flush;

    // Create a couply to try
    Vec3 test1(1, 2, 3);
    Vec3 test2(1, 2, 3);
    Vec3 test3(4, 5, 6);

    // Try some
    ASSERT(test1 == test2)
}


int main() {
    cout << "Testing Vec3-library..." << endl;

    if (!test_equal()) {
        return EXIT_FAILURE;
    }

    cout << "Done." << endl;
}

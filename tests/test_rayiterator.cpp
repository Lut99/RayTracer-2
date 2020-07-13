/* TEST RAYITERATOR.cpp
 *   by Lut99
 *
 * Created:
 *   09/07/2020, 17:29:20
 * Last edited:
 *   13/07/2020, 16:43:31
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This class tests the RayIterator. Specifically, it tries if it returns
 *   the expected rays in the expected order.
**/

#include <iostream>

#include "Vec3.hpp"
#include "Camera.hpp"
#include "RayIterator.hpp"

#include "Frame.hpp"

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


bool test_iteration() {
    // Create an example camera
    Camera cam(Point3(1, 2, 3), Point3(4, 5, 6), Vec3(0, 1, 0), 90, 500, 250);

    // Iterate over the possible rays, compare with the directly casted rays.
    RayIterator iter(cam, 1);
    size_t x = 0;
    size_t y = 0;
    for (Ray ray : iter) {
        Ray gold = cam.cast(x, y);
        if (ray != gold) {
            cout << "[FAIL]" << endl << endl;
            cerr << "ERROR: " << ray << " is not equal to " << gold << " @ (" << x << ", " << y << ")" << endl << endl;
            return false;
        }

        y += (x + 1) / 500;
        x = (x + 1) % 500;
    }

    cout << "[ OK ]" << endl;
    return true;
}


int main() {
    cout << "   Testing iteration correctness...           " << flush;
    test_iteration();

    cout << "Done." << endl << endl;
    return 0;
}

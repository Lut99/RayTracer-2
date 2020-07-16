/* TEST DEBUG.cpp
 *   by Lut99
 *
 * Created:
 *   16/07/2020, 16:04:27
 * Last edited:
 *   16/07/2020, 16:43:25
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file tests if the debug suite prints accurate errors.
**/

#include "Debug.hpp"

using namespace std;


namespace Test {
    void test1() {
        DEBUG_ENTER("test1");

        raise(SIGSEGV);

        DEBUG_RETURN();
    }
    void test2(int i = 0) {
        DEBUG_ENTER("test2");

        // Call test2 multiple times
        if (i < 16) {
            test2(i + 1);
        } else {
            test1();
        }

        DEBUG_RETURN();
    }
}
void test3() {
    DEBUG_ENTER("test3");

    Test::test2();

    DEBUG_RETURN();
}
void test4() {
    DEBUG_ENTER("test4");

    test3();

    DEBUG_RETURN();
}
void test5() {
    DEBUG_ENTER("test5");

    test4();

    DEBUG_RETURN();
}


int main() {
    DEBUG_INIT();

    // Let's make an error
    test5();

    // Done
    DEBUG_CLOSE();
    
    return 0;
}

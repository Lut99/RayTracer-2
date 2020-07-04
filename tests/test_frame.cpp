/* TEST FRAME.cpp
 *   by Lut99
 *
 * Created:
 *   7/4/2020, 12:08:25 PM
 * Last edited:
 *   7/4/2020, 3:10:07 PM
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
#include <cmath>

using namespace std;
using namespace RayTracer;


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
        p.b = 0;
    }

    // Write the image
    frame.toPNG("tests/test.png");

    cout << "[ OK ]" << endl;
    return true;
}


int main() {
    test_values();
    test_png();

    cout << "Done." << endl << endl;
    return EXIT_SUCCESS;
}

/* COLLECTFOUR.cpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 17:51:44
 * Last edited:
 *   13/07/2020, 18:08:49
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file is here to test the functionality of writing a character
 *   four-at-a-time as an integer to a buffer.
**/

#include <iostream>

using namespace std;

int main() {
    unsigned char buffer[1024];
    for (size_t i = 0; i < 1024; i++) { buffer[i] = 0; }

    // String to copy
    const char* str = "Hello there!\nGeneral Kenobi, you are a bold one...\n";

    // Copy it
    unsigned int to_write = 0;
    for (size_t i = 0; ; i++) {
        // Fetch the character
        char c = str[i];

        // Put at the correct place in the to_write int
        int index = i % 4;
        unsigned int value = c;
        to_write |= value << (8 * (3 - index));
        if (c == '\0' || index == 3) {
            // Send the index off the the buffer
            ((unsigned int*) buffer)[i / 4] = to_write;
            to_write = 0;
        }

        // Stop clause
        if (c == '\0') { break; }
    }

    // Print the buffer and see how it went
    cout << ((char*) buffer) << endl;
}

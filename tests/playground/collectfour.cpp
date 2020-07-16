/* COLLECTFOUR.cpp
 *   by Lut99
 *
 * Created:
 *   13/07/2020, 17:51:44
 * Last edited:
 *   16/07/2020, 14:42:44
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file is here to test the functionality of writing a character
 *   four-at-a-time as an integer to a buffer.
**/

#include <iostream>

using namespace std;


void write(unsigned char* buffer, size_t& buffer_i, const char* text) {
    unsigned int to_write = 0;
    // Loop and add each four characters
    for (; ; buffer_i++) {
        // Fetch the character
        char c = text[buffer_i];

        // Put at the correct place in the to_write int
        int index = buffer_i % 4;
        unsigned int value = (unsigned int) c;
        if (value == '\0') { value = (unsigned int) '\n'; }
        to_write |= value << (8 * index);
        if (c == '\0' || index == 3) {
            // Send the index off the the buffer
            ((unsigned int*) buffer)[buffer_i / 4] += to_write;
            to_write = 0;
        }

        // Stop clause
        if (c == '\0') { break; }
    }
}


int main(int argc, char** argv) {
    if (argc < 2) { cerr << "Usage: " << argv[0] << " [text*]" << endl; return EXIT_SUCCESS; }

    unsigned char buffer[1024];
    for (size_t i = 0; i < 1024; i++) { buffer[i] = 0; }

    // Copy it
    size_t i = 0;
    for (int j = 1; j < argc; j++) {
        write(buffer, i, argv[j]);
    }

    // Print the buffer and see how it went
    cout << ((char*) buffer);
}

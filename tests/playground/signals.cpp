/* SIGNALS.cpp
 *   by Lut99
 *
 * Created:
 *   16/07/2020, 14:59:28
 * Last edited:
 *   16/07/2020, 15:11:06
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file allows us to test signal handling
**/

#include <iostream>
#include <csignal>

using namespace std;


void segfault_handler(int) {
    cout << "My my! A segfault occured!" << endl;
    exit(EXIT_FAILURE);
}

void interrupt_handler(int) {
    cout << "My my! An interrupt occured!" << endl;
    exit(EXIT_FAILURE);
}


int main() {
    signal(SIGSEGV, segfault_handler);
    signal(SIGINT, interrupt_handler);

    raise(SIGSEGV);

    return EXIT_SUCCESS;
}

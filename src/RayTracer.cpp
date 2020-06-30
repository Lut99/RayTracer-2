/* RAY TRACER.cpp
 *   by Lut99
 *
 * Created:
 *   6/28/2020, 3:21:48 PM
 * Last edited:
 *   6/30/2020, 4:45:51 PM
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This is the entry point to the Raytracer. Here, we mostly parse the
 *   command line arguments and verify that they are valid, before moving
 *   to the correct renderer.
**/

#include <iostream>
#include <sstream>
#include <string>
#include <map>

#include "Renderer.hpp"

using namespace std;
using namespace RayTracer;



/***** CONSTANTS *****/
#ifndef RENDERER
#define RENDERER "sequential"
#endif



/***** DEFINITIONS *****/
/* Contains all the options that can be set by the ArgumentParser. */
struct Options {
    // TBD

    Options() {
        // TBD
    }
};



/***** HELPER FUNCTIONS *****/
/* Creates a copy of given string which is all lowercase. */
string tolower(const string& text) {
    stringstream result;
    for (char c : text) {
        result << (char) tolower(c);
    }
    return result.str();
}



/***** PARSING *****/
/* Prints a usage message to the given stream. */
ostream& print_usage(ostream& os, const string& execpath) {
    os << "Usage: " << execpath << " [-h]" << endl;
    return os;
}

/* Prints a help message to the given stream. */
ostream& print_help(ostream& os, const string& execpath) {
    print_usage(os, execpath);

    os << endl << "Miscellaneous:" << endl;
    os << "-h,--help\t\t\tShows this help menu." << endl;

    os << endl;
    return os;
}

/* Parses an argument as a label one. */
void parse_label(Options& result, const string& execpath, const string& label, const int remc, const char** remv) {
    // Stop if the label is empty
    if (label.empty()) {
        cerr << "ERROR: Cannot parse empty option" << endl;
        exit(EXIT_FAILURE);
    }

    // Switch to the correct label
    if (label[0] == 'r' || label == "-renderer") {
        // Parse the renderer used
        cout << "TBD" << endl;
    } else if (label[0] == 'h' || label == "-help") {
        // Print the help message
        print_help(cout, execpath);
        // Also, we're done
        exit(EXIT_SUCCESS);
    } else {
        cout << "??? : " << label << endl;
    }
}

/* Parses a positional according to where we are in the positional counts. */
void parse_positional(Options& result, const string& execpath, int position, const int remc, const char** remv) {
    
}

/* Parsers the command line arguments. */
Options parse_args(int argc, const char** argv) {
    // Loop through them all
    Options result;
    int positional_index = 0;
    string path(argv[0]);
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (arg[0] == '-') {
            // Parse as longlabel
            parse_label(result, path, tolower(string(arg + 1)), argc - i - 1, argv + i + 1);
        } else {
            // Parse as positional
            parse_positional(result, path, positional_index, argc - i - 1, argv + i + 1);
        }
    }
    return result;
}



/***** ENTRY POINT *****/
int main(int argc, const char** argv) {
    Options options = parse_args(argc, argv);

    cout << endl << "*** RAYTRACER v0.0.1 ***" << endl << endl;

    cout << "Settings:" << endl;
    cout << " - Rendering backend : " RENDERER << endl;
    cout << endl;

    // Run the renderer
    render();

    // Done
    cout << endl << "Done." << endl << endl;
    return 0;
}

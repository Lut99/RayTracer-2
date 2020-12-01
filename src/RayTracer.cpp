/* RAY TRACER.cpp
 *   by Lut99
 *
 * Created:
 *   6/28/2020, 3:21:48 PM
 * Last edited:
 *   17/07/2020, 17:49:54
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
#define RENDERER "seq"
#endif

#define DEFAULT_WIDTH 320
#define DEFAULT_HEIGHT 200
#define DEFAULT_VFOV 90
#define DEFAULT_UP Vec3(0, 1, 0)
#define DEFAULT_LOOKFROM Point3(0, 0, 0)
#define DEFAULT_LOOKAT Point3(1, 0, 0)


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
    os << "Usage: " << execpath << " [-hgp] [-whuFfa]" << endl;
    return os;
}

/* Prints a help message to the given stream. */
ostream& print_help(ostream& os, const string& execpath) {
    print_usage(os, execpath);

    os << endl << "Features:" << endl;

    os << endl << "Camera overrides:" << endl;
    os << "-W,--width\tDetermines the width (in pixels) of the first camera in the scene. (DEFAULT: " << DEFAULT_WIDTH << ")" << endl;
    os << "-H,--height\tDetermines the height (in pixels) of the first camera in the scene. (DEFAULT: " << DEFAULT_HEIGHT << ")" << endl;
    os << "-u,--up\t\tThe in-world vector over which the Y-axis of the Frame goes. (DEFAULT: " << DEFAULT_UP << ")" << endl;
    os << "-F,--vfov\tThe vertical field-of-view (in degrees) of the fist Camera. (DEFAULT: " << DEFAULT_VFOV << ")" << endl;
    os << "-f,--lookfrom\tThe coordinate of the point where the Camera looks from. (DEFAULT: " << DEFAULT_LOOKFROM << ")" << endl;
    os << "-a,--lookat\tThe coordinate of the point where the Camera looks to. (DEFAULT: " << DEFAULT_LOOKAT << ")" << endl;

    os << endl << "Post-processing:" << endl;
    os << "-g,--gamma\tIf given, corrects the gamme of the first Camera in the scene." << endl;

    os << endl << "Miscellaneous:" << endl;
    os << "-p,--progress\tIf given, shows a progressbar while rendering." << endl;
    os << "-h,--help\tShows this help menu and then exits." << endl;

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

    // Options
    if (label == "W" || label == "-width") {
        // Parse the next word as string
    } else if (label == "H" || label == "-height") {
        
    } else if (label == "u" || label == "-up") {
        
    } else if (label == "F" || label == "-vfov") {
        
    } else if (label == "f" || label == "-lookfrom") {
        
    } else if (label == "a" || label == "-lookat") {
        
    } else {
        for (char c : label) {
            // Flags
            if (c == 'g' || label == "-gamma") {
                
            } else if (c == 'p' || label == "-progress") {
                
            } else if (c == 'h' || label == "-help") {
                // Print the help message
                print_help(cout, execpath);
                // Also, we're done
                exit(EXIT_SUCCESS);
            }
        }
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

    // Create the renderer
    Renderer* renderer = rendererFactory();

    // Render the default scene
    Frame result = renderer->render()[0];
    result.toPNG("output/test.png");

    // Done
    cout << endl << "Done." << endl << endl;
    return 0;
}

/* RAY TRACER.cpp
 *   by Lut99
 *
 * Created:
 *   6/28/2020, 3:21:48 PM
 * Last edited:
 *   6/28/2020, 3:59:00 PM
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

using namespace std;


/***** DEFINITIONS *****/
/* Enum which determines the type of parser. */
enum class Parser {
    sequential,
    cuda
};

/* Contains all the options that can be set by the ArgumentParser. */
struct Options {
    Parser parser;

    Options() {
        // Set the default values
        this->parser = Parser::sequential;
    }
};



/***** HELPER FUNCTIONS *****/
/* Creates a copy of given string which is all lowercase. */
string tolower(const string& text) {
    stringstream result;
    for (char c : text) {
        result << tolower(c);
    }
    return result.str();
}



/***** PARSING *****/
/* Parses an argument as a label one. */
void parse_label(Options& result, const string& label, const int remc, const char** remv) {
    // Stop if the label is empty
    if (label.empty()) {
        cerr << "ERROR: Cannot parse empty option" << endl;
        exit(EXIT_FAILURE);
    }

    // Switch to the correct label
    if (label[0] == 'p' || label == "-parser") {
        // TBD
    }
}

/* Parses a positional according to where we are in the positional counts. */
void parse_positional(Options& result, int position, const int remc, const char** remv) {
    
}

/* Parsers the command line arguments. */
Options parse_args(int argc, const char** argv) {
    // Loop through them all
    Options result;
    int positional_index = 0;
    for (int i = 0; i < argc; i++) {
        const char* arg = argv[i];
        if (arg[0] == '-') {
            // Parse as longlabel
            parse_label(result, tolower(string(arg + 1)), argc - i - 1, argv + i + 1);
        } else {
            // Parse as positional
            parse_positional(result, positional_index, argc - i - 1, argv + i + 1);
        }
    }
    return result;
}



/***** ENTRY POINT *****/
int main(int argc, const char** argv) {
    Options options = parse_args(argc, argv);

    return 0;
}

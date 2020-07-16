/* DEBUG.hpp
 *   by Lut99
 *
 * Created:
 *   16/07/2020, 15:18:11
 * Last edited:
 *   16/07/2020, 16:56:23
 * Auto updated?
 *   Yes
 *
 * Description:
 *   This file contains a Debug suite, which can be used for certain assert
 *   statements to show a matching stacktrace. It automatically throws
 *   whenever C-style signals are thrown, and using the special DEBUG_THROW()
 *   macro, one can also see stacktraces upon seeing uncaught exceptions.
**/

#ifndef DEBUG_HPP
#define DEBUG_HPP

#include <cstdlib>
#include <cstddef>
#include <csignal>
#include <cstdio>


/***** CONSTANTS *****/

/* Number of frames maximally allowed on the stacktrace. */
#define STACKTRACE_SIZE 128



/***** NAMESPACE DECLARATIONS *****/

namespace Debug {
    /* A single frame in the stacktrace. */
    struct Frame {
        /* The line number where the current function is located. */
        size_t func_line;
        /* The name of the current file. */
        const char* file_name;
        /* The name (preferably header) of the current function. */
        const char* func_name;
        /* The human-readable context of the current function. */
        const char* context;
    };

    /* Stores the stacktrace that we will build. */
    __attribute__((unused)) static Frame* stacktrace[STACKTRACE_SIZE];
    /* Determines the used number of frames in the stacktrace. */
    __attribute__((unused)) static size_t stacktrace_size = 0;
    /* Keeps track of how many frames have been skipped. */
    __attribute__((unused)) static size_t skipped_frames;

    /* Returns the position of the last slash in a given string. Does not disambiguate between forward and backward slashes. */
    size_t find_last_slash(const char* text);

    /* Registers the relevant signal handlers. */
    void register_signals();
    /* Unregisters the relevant signal handlers. */
    void unregister_signals();
    /* Frees the memory of the stacktrace. */
    void free_stacktrace();

    /* Adds a frame with given number and function header. */
    void push_frame(size_t func_line, const char* file_name, const char* func_name, const char* context);
    /* Removes the last pushed frame from the stacktrace and returns it. */
    Frame* pop_frame();
    /* Does not remove but only returns the last pushed frame from the stacktrace. */
    Frame* peek_frame();

    /* Prints the stacktrace. */
    void print_trace(FILE* file);

    /***** SIGNAL HANDLERS *****/
    
    /* Handles SIGSEGV. */
    void handle_SIGSEGV(int);
};



#ifdef DEBUG
/***** DEBUG MACROS *****/

/*** DEBUG CLASS META ***/

/* Initializes the suite and immediate stacktraces the calling function. */
#define DEBUG_INIT() \
    Debug::register_signals(); \
    DEBUG_ENTER("main");

/* De-initializes the suite. */
#define DEBUG_CLOSE() \
    Debug::free_stacktrace(); \
    Debug::unregister_signals();



/*** STACKTRACE ***/

/* Enters a function, and sets the function name / context description for use in other macros. */
#define DEBUG_ENTER(CONTEXT) \
    __attribute__((unused)) const char* __DEBUG_CONTEXT__ = CONTEXT; \
    Debug::push_frame(__LINE__ - 1, __FILE__, __PRETTY_FUNCTION__, CONTEXT);

/* Leaves a function, removing it from the stacktrace. */
#define DEBUG_RETURN(VALUE) \
    delete Debug::pop_frame(); \
    return VALUE;



/***** IMPLEMENTATIONS *****/

/* Returns the position of the last slash in a given string. Does not disambiguate between foraard and backward slashes. */
size_t Debug::find_last_slash(const char* text) {
    size_t last_index = 0;
    for(size_t i = 0; ; ++i) {
        char c = text[i];

        // Switch to do the correct action after find a certain char
        switch(c) {
            case '/':
                last_index = i;
                break;
            case '\\':
                last_index = i;
                break;
            case '\0':
                return last_index;
            default:
                break;
        }
    }
    return last_index;
}

/* Registers the relevant signal handlers. */
void Debug::register_signals() {
    // Register SIGSEGV
    signal(SIGSEGV, handle_SIGSEGV);
}

/* Unregisters the relevant signal handlers. */
void Debug::unregister_signals() {
    // Unregister SIGSEGV
    signal(SIGSEGV, SIG_DFL);
}

/* Frees all elements of the stacktrace. */
void Debug::free_stacktrace() {
    while (stacktrace_size > 0) {
        delete pop_frame();
    }
}

/* Adds a frame with given number and function header. */
void Debug::push_frame(size_t func_line, const char* file_name, const char* func_name, const char* context) {
    using namespace Debug;

    if (stacktrace_size == STACKTRACE_SIZE - 1) {
        // Make new space first by moving everything one back, deleting the first
        delete stacktrace[0];
        for (size_t i = 1; i < stacktrace_size; i++) {
            stacktrace[i - 1] = stacktrace[i];
        }
        // Decrement the stacktrace size, increment the skipped
        stacktrace_size--;
        skipped_frames++;
    }

    // Create a new frame at the end of the stack
    stacktrace[stacktrace_size++] = new Frame({func_line, file_name, func_name, context});
}

/* Removes the last pushed frame from the stacktrace. */
Debug::Frame* Debug::pop_frame() {
    using namespace Debug;

    // Remove the last frame if there is any to remove
    if (stacktrace_size > 0) {
        return stacktrace[--stacktrace_size];
    }

    // Else, return nullptr
    return nullptr;
}

Debug::Frame* Debug::peek_frame() {
    using namespace Debug;

    // Remove the last frame if there is any to remove
    if (stacktrace_size > 0) {
        return stacktrace[stacktrace_size - 1];
    }

    // Else, return nullptr
    return nullptr;
}

/* Prints the stacktrace. */
void Debug::print_trace(FILE* file) {
    using namespace Debug;

    fprintf(file, "Stacktrace:\n");
    // Write the elements of the stacktrace
    while (stacktrace_size > 0) {
        // Print the frame
        Frame* frame = pop_frame();
        fprintf(file, "   at `%s` [\"%s\":%lu]\n", frame->func_name, frame->file_name + find_last_slash(frame->file_name) + 1, frame->func_line);
        delete frame;
    }

    // Finally, print any unshown stacktraces
    if (skipped_frames > 0) {
        fprintf(file, "   ... %lu more\n", skipped_frames);
    }
}



/* Handles SIGSEGV. */
void Debug::handle_SIGSEGV(int errno) {
    using namespace Debug;

    // Write the general error
    fprintf(stderr, "\nERROR: %s: Segmentation fault.\n", peek_frame()->context);
    
    print_trace(stderr);
    fprintf(stderr, "\n");
    
    exit(errno);
}

#else
/***** DEBUG MACROS *****/

/*** DEBUG CLASS META ***/

/* Initializes the suite. */
#define DEBUG_INIT()

/* De-initializes the suite. */
#define DEBUG_CLOSE()



/*** STACKTRACE ***/

/* Enters a function, and sets the function name / context description for use in other macros. */
#define DEBUG_ENTER(CONTEXT)

/* Leaves a function, removing it from the stacktrace. */
#define DEBUG_RETURN(VALUE) \
    return VALUE;



/***** IMPLEMENTATIONS *****/

/* Returns the position of the last slash in a given string. Does not disambiguate between foraard and backward slashes. */
size_t Debug::find_last_slash(const char* text) { return 0; }
/* Registers the relevant signal handlers. */
void Debug::register_signals() {  /* Does nothin' */ }
/* Unregisters the relevant signal handlers. */
void Debug::unregister_signals() { /* Does nothin' */ }
/* Frees all elements of the stacktrace. */
void Debug::free_stacktrace() { /* Does nothin' */ }
/* Adds a frame with given number and function header. */
void Debug::push_frame(size_t, const char*, const char*, const char*) { /* Does nothin' */ }
/* Removes the last pushed frame from the stacktrace. */
Debug::Frame* Debug::pop_frame() { return nullptr; }
/* Does not remove but only returns the last pushed frame from the stacktrace. */
Debug::Frame* Debug::peek_frame() { return nullptr; }
void Debug::print_trace(FILE*) { /* Does nothin' */ }

/* Handles SIGSEGV. */
void Debug::handle_SIGSEGV(int) { /* Does nothin' */ }

#endif

#endif

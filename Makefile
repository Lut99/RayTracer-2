# MAKEFILE for the RAYTRACER-2 PROJECT
#   by Lut99
#

### CONSTANTS ###
GXX=g++
GXX_ARGS=-O2 -Wall -Wextra -std=c++17

SRC=src
LIB=$(SRC)/lib
INCL=-I$(LIB)/include

BIN=bin
OBJ=bin/obj


### DEFINE LIBRARIES ###


### PROCESS INPUT ###


### PHONY RULES ###
.PHONY: default all clean
default: all

all: raytracer
clean:
	rm -f $(OBJ)/*.o
	rm -r $(BIN)/*

raytracer: $(BIN)/raytracer

### FILE RULES ###
$(OBJ)/RayTracer.o: $(SRC)/RayTracer.cpp
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(BIN)/raytracer: $(OBJ)/RayTracer.o
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ $^

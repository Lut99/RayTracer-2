# MAKEFILE for the RAYTRACER-2 PROJECT
#   by Lut99
#

### CONSTANTS ###
GXX=g++
GXX_ARGS=-O2 #-Wall -Wextra -std=c++17

SRC=src
LIB=$(SRC)/lib

BIN=bin
OBJ=bin/obj

RENDERER_FILE=$(OBJ)/renderers/SequentialRenderer.o

### DEFINE LIBRARIES ###
LIBS := $(shell find $(LIB) -name '*.cpp')
LIBS := $(LIBS:$(LIB)/%.cpp=$(OBJ)/%.o )
DIRS := $(sort $(dir $(wildcard $(LIB)/*/)))
INCL := $(DIRS:%=-I% )
DIRS := $(DIRS:$(LIB)/%=$(OBJ)/% )

### PROCESS INPUT ###
ifdef DEBUG
GXX_ARGS += -g
endif

ifdef RENDERER
ifneq ($(RENDERER), seq)
ifneq ($(RENDERER), cpu)
ifneq ($(RENDERER), cuda)
$(error Unknown RENDERER (must be 'seq', 'cpu' or 'cuda'))
endif
endif
endif
GXX_ARGS += -DRENDERER=\"$(RENDERER)\"
# Remove all renderers except the chosen one
LIBS := $(LIBS:$(OBJ)/renderers/%.o= )
LIBS := $(LIBS) $(OBJ)/renderers/$(RENDERER).o
else
# Remove all renderers except seq.o
LIBS := $(LIBS:$(OBJ)/renderers/%.o= )
LIBS := $(LIBS) $(OBJ)/renderers/seq.o
endif


### PHONY RULES ###
.PHONY: default all clean dirs raytracer
default: all

all: raytracer
clean:
	rm -f $(OBJ)/*.o
	rm -f $(BIN)/*.out

dirs: $(BIN) $(OBJ) $(DIRS)

raytracer: $(BIN)/raytracer.out

### DIRECTORY RULES ###
$(BIN):
	mkdir -p $@
$(OBJ):
	mkdir -p $@
$(OBJ)/%/:
	mkdir -p $@

### FILE RULES ###
$(OBJ)/%.o: $(LIB)/%.cpp | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(OBJ)/%.o: $(LIB)/%.cu | dirs
	$(info TBD)

$(OBJ)/RayTracer.o: $(SRC)/RayTracer.cpp | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(BIN)/raytracer.out: $(OBJ)/RayTracer.o $(LIBS) | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ $^

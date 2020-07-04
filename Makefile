# MAKEFILE for the RAYTRACER-2 PROJECT
#   by Lut99
#

### CONSTANTS ###
GXX=g++
GXX_ARGS=-O2 -Wall -Wextra -std=c++17
NVCC=nvcc
NVCC_ARGS=-O2 --gpu-architecture=compute_75 --gpu-code=sm_75

EXT_LIBS=-lm

SRC=src
LIB=$(SRC)/lib

BIN=bin
OBJ=$(BIN)/obj

TST_SRC=tests
TST_BIN=$(BIN)/tests

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

# CUDA specific shit:
ifeq ($(RENDERER), cuda)
# Replace our own libraries by their CUDA counterpart
LIBS := $(LIBS:$(OBJ)/math/Vec3.o=$(OBJ)/math/GVec3.o )

# Modify the external libraries
EXT_LIBS += -L/usr/local/cuda-11.0/lib64 -lcuda -lcudart
endif

else
# Remove all renderers except seq.o
LIBS := $(LIBS:$(OBJ)/renderers/%.o= )
LIBS := $(LIBS) $(OBJ)/renderers/seq.o
endif


### PHONY RULES ###
.PHONY: default all clean dirs raytracer tests
default: all

all: raytracer
clean:
	find $(OBJ) -name "*.o" -type f -delete
	rm -f $(BIN)/*.out

dirs: $(BIN) $(TST_BIN) $(OBJ) $(DIRS)

raytracer: $(BIN)/raytracer.out

tests: $(TST_BIN)/test_vec3.out $(TST_BIN)/test_gvec3.out $(TST_BIN)/test_frame.out $(TST_BIN)/test_gframe.out
	$(info Running tests...)
	$(info )

	$(TST_BIN)/test_vec3.out
	$(TST_BIN)/test_gvec3.out
	$(TST_BIN)/test_frame.out
	$(TST_BIN)/test_gframe.out

### DIRECTORY RULES ###
$(BIN):
	mkdir -p $@
$(TST_BIN):
	mkdir -p $@
$(OBJ):
	mkdir -p $@
$(OBJ)/%/:
	mkdir -p $@

### FILE RULES ###
$(OBJ)/%.o: $(LIB)/%.cpp | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(OBJ)/%.o: $(LIB)/%.cu | dirs
	$(NVCC) $(NVCC_ARGS) $(INCL) -o $@ -dc $<

$(OBJ)/RayTracer.o: $(SRC)/RayTracer.cpp | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(BIN)/raytracer.out: $(OBJ)/RayTracer.o $(LIBS) | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ $^ $(EXT_LIBS)


### TEST RULES ###
$(OBJ)/test_%.o: $(TST_SRC)/test_%.cpp | dirs
	$(GXX) $(GXX_ARGS) $(INCL) -o $@ -c $<

$(OBJ)/test_%.o: $(TST_SRC)/test_%.cu | dirs
	$(NVCC) $(NVCC_ARGS) $(INCL) -o $@ -dc $<

$(TST_BIN)/test_vec3.out: $(OBJ)/test_vec3.o $(OBJ)/math/Vec3.o | dirs
	$(GXX) $(GXX_ARGS) -o $@ $^ $(EXT_LIBS)

$(TST_BIN)/test_gvec3.out: $(OBJ)/test_gvec3.o $(OBJ)/math/Vec3.o $(OBJ)/math/GVec3.o | dirs
	$(NVCC) $(NVCC_ARGS) -o $@ $^ $(EXT_LIBS)

$(TST_BIN)/test_frame.out: $(OBJ)/test_frame.o $(OBJ)/frames/Frame.o $(OBJ)/frames/LodePNG.o | dirs
	$(GXX) $(GXX_ARGS) -o $@ $^ $(EXT_LIBS)

$(TST_BIN)/test_gframe.out: $(OBJ)/test_gframe.o $(OBJ)/frames/GFrame.o $(OBJ)/frames/Frame.o $(OBJ)/frames/LodePNG.o | dirs
	$(NVCC) $(NVCC_ARGS) -o $@ $^ $(EXT_LIBS)
	# $(GXX) $(GXX_ARGS) -o $@ $^ $(EXT_LIBS) -L/usr/local/cuda-11.0/lib64 -lcuda -lcudart

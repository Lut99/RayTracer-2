.PHONY: default all cpugpuhybrid dirs
default: all

all: cpugpuhybrid

### DIR RULES ###
bin:
	mkdir -p $@

dirs: bin

### CPU GPU HYBRID ###
bin/cpugpuhybrid.out: cpugpuhybrid.cu | dirs
	nvcc -O2 --gpu-architecture=compute_75 --gpu-code=sm_75 -o $@ $^
cpugpuhybrid: bin/cpugpuhybrid.out

### GPU COPY ###
bin/gpucopy.out: gpucopy.cu | dirs
	nvcc -O2 --gpu-architecture=compute_75 --gpu-code=sm_75 -o $@ $^
gpucopy: bin/gpucopy.out

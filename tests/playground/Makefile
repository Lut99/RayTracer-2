.PHONY: default all cpugpuhybrid gpucopy collectfour signals dirs
default: all

all: cpugpuhybrid gpucopy collectfour signals

clean:
	rm -f bin/*.out

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

### COLLECT FOUR ###
bin/collectfour.out: collectfour.cpp | dirs
	g++ -std=c++17 -O2 -Wall -Wextra -o $@ $^
collectfour: bin/collectfour.out

### SIGNALS ###
bin/signals.out: signals.cpp | dirs
	g++ -std=c++17 -O2 -Wall -Wextra -o bin/signals.out signals.cpp
signals: bin/signals.out

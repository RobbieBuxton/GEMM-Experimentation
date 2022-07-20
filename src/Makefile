.DEFAULT_GOAL := run

CC=gcc
CFLAGS=  -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -fopenmp
LIBS= -lblas

run: clean test
	./test

test: test.o utils.o kernels/stencil/openblas.o kernels/stencil/devito.o kernels/stencil/custom.o
	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean: 
	rm -rf test.o utils.o kernels/stencil/openblas.o kernels/stencil/devito.o kernels/stencil/custom.o
.DEFAULT_GOAL := test

CC=gcc
CFLAGS= -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -fopenmp
LIBS= -lopenblas -lm
ENV = OPENBLAS_NUM_THREADS=6
graph1: clean test
	./test 9 250 250 250 250 0.6 0.1 0.1 0.1 0.1

graph2: clean test
	./test 49 0 2000 25 25 0.6 0.1 0.1 0.1 0.1

test: test.o utils.o kernels/stencil/openblas.o kernels/stencil/devito.o kernels/stencil/matrix_helpers.o kernels/stencil/custom.o
	${ENV} $(CC) -o  $@ $^ $(CFLAGS) $(LIBS)

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

clean: 
	rm -rf ./test test.o utils.o kernels/stencil/openblas.o kernels/stencil/devito.o kernels/stencil/custom.o kernels/stencil/matrix_helpers.o
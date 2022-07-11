.DEFAULT_GOAL := build
CFLAGS = -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -fopenmp 
LIBS = -lblas
CC = gcc

build: src/test.o src/utils.o src/kernels/gemm.o src/kernels/devito.o
	@echo "Building ..." 
	$(CC) -o mult src/test.o src/kernels/gemm.o src/utils.o src/kernels/devito.o $(CFLAGS) $(LIBS) 

	@echo "Finished ..."

run: clean build
	./mult	

clean:
	@echo "Cleaning up..."
	rm mult
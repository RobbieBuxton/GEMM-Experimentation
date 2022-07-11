.DEFAULT_GOAL := build
CFLAGS = -O3 -g -fPIC -Wall -std=c99 -march=native -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -fopenmp 
LIBS = -lblas
CC = gcc

build: src/test.o src/utils.o
	@echo "Building ..." 
	$(CC) -o mult src/test.o src/utils.o $(CFLAGS) $(LIBS) 

	@echo "Finished ..."

run: build
	./mult	

clean:
	@echo "Cleaning up..."
	rm mult
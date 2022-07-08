.DEFAULT_GOAL := build

build:
	@echo "Building ..."
	gcc -O3 -g -fPIC -Wall -std=c99 -lblas -march=native -Wno-unused-result -Wno-unused-variable -Wno-unused-but-set-variable -ffast-math -fopenmp src/test.c src/utils.c -lblas -o mult
	@echo "Finished ..."

run: build
	./mult	

clean:
	@echo "Cleaning up..."
	rm mult
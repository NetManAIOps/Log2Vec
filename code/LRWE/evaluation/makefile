CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
all: Distance Compute-accuracy
Distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
Compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
clean:
	rm -rf distance compute-accuracy

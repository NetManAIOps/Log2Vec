CC = g++
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result
all: Word2vec LRCWE
Word2vec: word2vec.c
	$(CC) word2vec.c -o word2vec $(CFLAGS)
LRCWE: lrcwe.c
	$(CC) lrcwe.c -o lrcwe $(CFLAGS)
clean:
	rm -rf word2vec lrcwe

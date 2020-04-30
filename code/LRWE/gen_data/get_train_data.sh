#!/bin/bash
if [ ! -f enwik9 ]; then
    wget http://cs.fit.edu/~mmahoney/compression/enwik9.zip
    unzip enwik9.zip
fi
./wikifil.pl enwik9 > train.txt

#!/bin/bash

rm -f *.csv
for i in 1000 5000 10000 30000; do
	for j in 4 8 16 32 64 128 256 512 1024; do
		echo "Testing MxN: $i x $i, block_size: $j..."
		echo "/*--------------------------------------------*/"
		./prog -n $i -m $i -b $j -t -w > /dev/null
	done
done

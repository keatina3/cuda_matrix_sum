#compilers
CC = gcc
CXX = gcc
NVCC = nvcc
LINK = nvcc

#flags 
CFLAGS = -W -Wall
NVCCFLAGS = -g -G --use_fast_math

INCPATH = /usr/include/

#files
OBJECTS = main.o
CU_OBJECTS = matrix_gpu.o
CU_SOURCES = matrix_gpu.cu

TARGET = prog

all: $(OBJECTS) cu
	$(NVCC) $(OBJECTS) $(CU_OBJECTS) -o $(TARGET) -I$(INCPATH)

cu: $(CU_OBJECTS)
	$(NVCC) $(CU_SOURCES) -c $(NVCCFLAGS) -I$(INCPATH)

.PHONY: clean

clean:
	$(RM) $(OBJECTS) $(TARGET)

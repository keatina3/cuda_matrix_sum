#compilers
CC = gcc
NVCC = nvcc

#flags 
CFLAGS = -W -Wall
NVCCFLAGS = -g -G --use_fast_math

#INCPATH = /usr/include/

#files
OBJECTS = main.o matrix.o
CU_OBJECTS = matrix_gpu.o
CU_SOURCES = matrix_gpu.cu

TARGET = prog

all: $(OBJECTS) cu_objs
	$(NVCC) $(OBJECTS) $(CU_OBJECTS) -o $(TARGET) #-I$(INCPATH)

cu_objs: $(CU_SOURCES)
	$(NVCC) $(CU_SOURCES) -c $(NVCCFLAGS) #-I$(INCPATH)

.PHONY: clean

clean:
	$(RM) $(OBJECTS) $(CU_OBJECTS) $(TARGET)

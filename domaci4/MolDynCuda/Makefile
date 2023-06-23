# Makefile for the molecular dynamics code

#
# C compiler and options for GNU 
#
CC=nvcc
CC_FLAGS = -O3 

#
# Object files
#
OBJ=    main.o \
	dfill.o \
	domove.o \
        dscal.o \
	fcc.o \
	forces.o \
	forcesGPU.o \
	mkekin.o \
	mxwell.o \
	prnout.o \
	velavg.o

CU= main.cu \
	dfill.cu \
	domove.cu \
        dscal.cu \
	fcc.cu \
	forces.cu \
	mkekin.cu \
	mxwell.cu \
	prnout.cu \
	velavg.cu

#
# Compile
#
main.o:	  main.cu
	 nvcc -c main.cu -o main.o


forcesGPU.o:  forcesGPU.cu
	 nvcc -g -G  -dc --maxrregcount=64  -arch=sm_61 -c forcesGPU.cu -o forcesGPU.o
#	nvcc -arch=sm_61 -c forcesGPU.cu -o forcesGPU.o

md:	$(OBJ)
	$(CC) $(CC_FLAGS) -o $@ $(OBJ) -lm


%.o: %.cu
	nvcc -c $< -o $@

#
# Clean out object files and the executable.
#
clean:
	rm *.o md

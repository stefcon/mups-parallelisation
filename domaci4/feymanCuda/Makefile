# C compiler
CC = nvcc
CC_FLAGS = -O3 --expt-relaxed-constexpr -rdc=true -lcudadevrt

feyman: 
	$(CC) $(CC_FLAGS) feyman.cu -o feyman 

feyman_f: 
	$(CC) $(CC_FLAGS) feyman_float.cu -o feyman 

clean:
	rm -f feyman

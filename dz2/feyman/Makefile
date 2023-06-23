# C compiler
CC = mpicc
CC_FLAGS = -O3

prime: 
	$(CC) $(CC_FLAGS) feyman.c -lm -o feyman 
prime_mod: 
	$(CC) $(CC_FLAGS) feyman_mod.c -lm -o feyman_mod 

clean:
	rm -f feyman
clean_mod:
	rm -f feyman_mod


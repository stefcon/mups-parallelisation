# C compiler
CC = mpicc
CC_FLAGS = -fopenmp -O3

prime: 
	$(CC) $(CC_FLAGS) prime.c -o prime
prime_mod: 
	$(CC) $(CC_FLAGS) prime_mod.c -lm -o prime_mod 
prime_base:
	$(CC) $(CC_FLAGS) prime_base.c -lm -o prime_base 

clean:
	rm -f prime
clean_mod:
	rm -f prime_mod
clean_base:
	rm -f prime_base

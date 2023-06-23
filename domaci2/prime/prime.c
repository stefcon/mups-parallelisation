#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>


double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

#define MASTER 0

int prime_number(int n)
{
  int i;
  int j;
  int prime;
  int total, total_master = 0;

  int size;
  int start, end, rank, chunk;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  
  chunk = (n + size - 1) / size;
  start = rank * chunk + 2;
  end = start + chunk - 1 < n ? start + chunk - 1 : n;
  if(end != start && end == n-1) end = n;
  total = 0;
  
  for (i = start; i <= end; i++)
  {
    prime = 1;
    for (j = 2; j * j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break; 
      }
    }
    total = total + prime;
  }

  MPI_Reduce(&total, &total_master, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);
  return total_master;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  size_t len;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  len = strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

void test(int n_lo, int n_hi, int n_factor);



int main(int argc, char *argv[])
{
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
  int n_factor;
  int n_hi;
  int n_lo;
  int i;
  int n;
  int primes;
  double ctime;
  //printf("RANK: %d \n", rank);
  if(rank==MASTER){
    timestamp();
    printf("\n");
    printf("PRIME TEST\n");
    
    if (argc != 4)
    {
      n_lo = 1;
      n_hi = 131072;
      n_factor = 2;
    }
    else
    {
      n_lo = atoi(argv[1]);
      n_hi = atoi(argv[2]);
      n_factor = atoi(argv[3]);
    }

    

    printf("\n");
    printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
    printf("\n");
    printf("         N        Pi          Time\n");
    printf("\n");

    n = n_lo;
  }
  MPI_Bcast(&n_hi, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  while (n <= n_hi)
  {
    if(rank==MASTER) 
      ctime = cpu_time();
    
    primes = prime_number(n);

    if(rank==MASTER) ctime = cpu_time() - ctime;

    if(rank==MASTER) printf("  %8d  %8d  %14f\n", n, primes, ctime);
    if(rank==MASTER) n = n * n_factor;
    
    MPI_Bcast(&n, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  }
  //printf("KRAJ: %d %d %d",rank, n, n_hi);
  if(rank==MASTER){
    printf("\n");
    printf("PRIME_TEST\n");
    printf("  Normal end of execution.\n");
    printf("\n");
    timestamp();
  }
  MPI_Finalize();
  return 0;
}

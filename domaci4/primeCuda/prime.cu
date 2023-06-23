#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#define NUM_OF_GPU_THREADS 1024

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

int prime_number(int n)
{
  int i;
  int j;
  int prime;
  int total;

  total = 0;

  for (i = 2; i <= n; i++)
  {
    prime = 1;
    for (j = 2; j < i; j++)
    {
      if ((i % j) == 0)
      {
        prime = 0;
        break;
      }
    }
    total = total + prime;
  }
  return total;
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
void testGPU(int n_lo, int n_hi, int n_factor);

__global__ void prime_number_kernel(int *d_total, int n)
{ 
  extern __shared__ int sdata[];

  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;
  if (i+2 <= n) {
    sdata[tid] = 1;
    for (int j = 2; j*j <= i+2; j++)
    {
      // Removing divergence
      sdata[tid] &= (bool) ((i+2) % j);
    }
  }
  else sdata[tid] = 0;
  // Block until all threads in the block have written their data to shared mem
  __syncthreads();
  // Do reduction in shared memory
  for (int iter = blockDim.x / 2; iter > 0; iter >>= 1) {
    if (tid < iter) {
      sdata[tid] = sdata[tid] + sdata[tid + iter];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) d_total[blockIdx.x] = sdata[0]; // Because of the inclusion of 1 for the first block
}

// CUDA main function
int prime_number_GPU(int n) {

  // define grid and block size
  const int numThreadsPerBlock = NUM_OF_GPU_THREADS;

  // Compute number of blocks needed based on array size and desired block size
  int numBlocks = ceil(1.0 * n / numThreadsPerBlock);

  // Events for time measurement - potentially
  // cudaEvent_t start, stop;
  // cudaEventCreate(&start);
  // cudaEventCreate(&stop);

  int sharedMemSize = numThreadsPerBlock * sizeof(int);
  int memSizeTotal = numBlocks * sizeof(int);

  int *total, *d_total;

  total = (int*)malloc(memSizeTotal);
  cudaMalloc(&d_total, memSizeTotal);

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);

  // cudaEventRecord(start);
  prime_number_kernel<<< dimGrid, dimBlock, sharedMemSize >>>( d_total, n );
  // cudaEventRecord(stop);

  // device to host copy
  cudaMemcpy( total, d_total, memSizeTotal, cudaMemcpyDeviceToHost );

  // cudaEventSynchronize(stop);
  // cudaEventElapsedTime(gtime, start, stop);

  // Final reduction
  int result = 0;
  for (int i = 0; i < memSizeTotal/sizeof(int); ++i) {
    result += total[i];
  }

  cudaFree(d_total);

  free(total);
  
  return result;
}

int main(int argc, char *argv[])
{
  int n_factor;
  int n_hi;
  int n_lo;

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
  printf("PRIME_TEST_CUDA\n");

  testGPU(n_lo, n_hi, n_factor);

  printf("\n");
  printf("PRIME_TEST_CUDA\n");
  printf("  Normal end of execution.\n");
  printf("\n");
  timestamp();

  return 0;
}

void test(int n_lo, int n_hi, int n_factor)
{
  int n;
  int primes;
  double ctime;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi          Time\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    ctime = cpu_time();

    primes = prime_number(n);

    ctime = cpu_time() - ctime;

    printf("  %8d  %8d  %14f\n", n, primes, ctime);
    n = n * n_factor;
  }

  return;
}

void testGPU(int n_lo, int n_hi, int n_factor)
{
  int n;
  int primes_cpu, primes_gpu;
  double ctime, gtime;
  bool failed;

  printf("\n");
  printf("  Call PRIME_NUMBER to count the primes from 1 to N.\n");
  printf("\n");
  printf("         N        Pi_C        Pi_G          C_T          G_T\n");
  printf("\n");

  n = n_lo;

  while (n <= n_hi)
  {
    // To initialize context for the first test case
    cudaDeviceSynchronize();

    ctime = cpu_time();

    primes_cpu = prime_number(n);

    ctime = cpu_time() - ctime;    

    gtime = cpu_time();

    primes_gpu = prime_number_GPU(n);

    gtime = cpu_time() - gtime;  

    printf("%8d  %8d  %8d  %14f  %14f\n", n, primes_cpu, primes_gpu, ctime, gtime);
    n = n * n_factor;
  }

  return;
}
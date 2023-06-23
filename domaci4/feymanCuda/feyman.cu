#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define NUM_OF_THREADS 1024
#define SUB_BLOCK 1024
#define A 3.0
#define B 2.0
#define C 1.0
#define ACCURACY 0.01
#define STEPSZ 0.0547722558
#define H 0.001

double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}

int i4_ceiling(double x)
{
  int value = (int)x;
  if (value < x)
    value = value + 1;
  return value;
}

int i4_min(int i1, int i2)
{
  int value;
  if (i1 < i2)
    value = i1;
  else
    value = i2;
  return value;
}

__host__ double potential(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

__device__ double potentialCuda(double x, double y, double z)
{
  return 2.0 * ((x / A / A)*(x / A / A) + (y / B / B)*(y / B / B) + (z / C / C)*(z / C / C)) + 1.0 / A / A + 1.0 / B / B + 1.0 / C / C;
}

__host__ __device__ double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  // if (*seed < 0)
  // {
  //   *seed = *seed + 2147483647;
  // }

  *seed += !((bool)max(*seed, 0)) * 2147483647;
  // *seed += (*seed < 0) * 2147483647;

  r = (double)(*seed) * 4.656612875E-10;

  return r;
}

void timestamp(void)
{
#define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time(NULL);
  tm = localtime(&now);

  strftime(time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm);

  printf("%s\n", time_buffer);

  return;
#undef TIME_SIZE
}

// print na stdout upotrebiti u validaciji paralelnog resenja
double testSeq(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  int dim = 3;
  double err;
  double h = 0.001;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps_ave;
  double w_exact;
  double wt;

  int N = atoi(argv[1]);
  double time = cpu_time();
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c)){
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c)){
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else{
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;
 
    for (int i = 1; i <= ni; i++){
        for (int j = 1; j <= nj; j++){
        for (int k = 1; k <= nk; k++){
        double x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);
        double y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);
        double z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        double chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
        w_exact = 1.0;
        wt = 1.0;
        steps_ave = 0;
            //printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
            //       x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

        continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        int steps = 0;
            
        double x1,x2,x3,ut,us,vs,vh,we,w,dx,dy,dz;
        
        for (int trial = 0; trial < N; trial++){
            x1 = x;
            x2 = y;
            x3 = z;
            w = 1.0;
            chk = 0.0;
            while (chk < 1.0){
            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
                us = r8_uniform_01(&seed) - 0.5;
                if (us < 0.0)
                dx = -stepsz;
                else
                dx = stepsz;
            }
            else
                dx = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
                us = r8_uniform_01(&seed) - 0.5;
                if (us < 0.0)
                dy = -stepsz;
                else
                dy = stepsz;
            }
            else
                dy = 0.0;

            ut = r8_uniform_01(&seed);
            if (ut < 1.0 / 3.0)
            {
                us = r8_uniform_01(&seed) - 0.5;
                if (us < 0.0)
                dz = -stepsz;
                else
                dz = stepsz;
            }
            else
                dz = 0.0;

            vs = potential(a, b, c, x1, x2, x3);
            x1 = x1 + dx;
            x2 = x2 + dy;
            x3 = x3 + dz;

            steps++;

            vh = potential(a, b, c, x1, x2, x3);

            we = (1.0 - h * vs) * w;
            w = w - 0.5 * h * (vh * we + vs * w);

            chk = pow(x1 / a, 2) + pow(x2 / b, 2) + pow(x3 / c, 2);
            }
            wt = wt + w;
        }
        
            wt = wt / (double)(N);
            steps_ave = steps / (double)(N);

            err = err + pow(w_exact - wt, 2);

            // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
            //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
        }
    }
}
  err = sqrt(err / (double)(n_inside));
  time = cpu_time() - time;
  printf("\nRMS absolute error in solution = %e\n", err);
  printf("CPU Time = %f\n", time);
  timestamp();
  printf("\n\n");

  return err;
}

__global__ void feyman_kernel(double *cu_points, double* cu_reduction_mem, int N) {

  extern __shared__ double sdata[];

  int ind = blockIdx.x*4;
  int tid = threadIdx.x;
  double x1 = cu_points[ind];
  double x2 = cu_points[ind + 1];
  double x3 = cu_points[ind + 2];
  double w_exact = cu_points[ind + 3];

  int seed = 123456789 - blockIdx.x * threadIdx.x;

  double vs,vh,w,dx,dy,dz,chk,us, ut;
  bool x_r, y_r, z_r;

  w = 1.0;
  chk = 0.0;
  sdata[tid] = 0.0;
  for (int trial = tid; trial < N; trial += blockDim.x){
    while (chk < 1.0){
        // Solution 1.
        // x_r = (bool) max(r8_uniform_01(&seed) - 0.5, 0.0);
        // y_r = (bool) max(r8_uniform_01(&seed) - 0.5, 0.0);
        // z_r = (bool) max(r8_uniform_01(&seed) - 0.5, 0.0);

        // dx = !(bool)max(r8_uniform_01(&seed) - 1./3., 0.0) *
        // (!x_r*-STEPSZ + x_r*STEPSZ);

        // dy = !(bool)max(r8_uniform_01(&seed) - 1./3., 0.0) *
        // (!y_r*-STEPSZ + y_r*STEPSZ); 

        // dz = !(bool)max(r8_uniform_01(&seed) - 1./3., 0.0) *
        // (!z_r*-STEPSZ + z_r*STEPSZ);

        // Solution 2.
        // x_r = (r8_uniform_01(&seed) - 0.5 < 0);
        // y_r = (r8_uniform_01(&seed) - 0.5 < 0);
        // z_r = (r8_uniform_01(&seed) - 0.5 < 0);

        // dx = (r8_uniform_01(&seed) < 1./3.) *
        // (!x_r*-STEPSZ + x_r*STEPSZ);

        // dy = (r8_uniform_01(&seed) < 1./3.) *
        // (!y_r*-STEPSZ + y_r*STEPSZ); 

        // dz = (r8_uniform_01(&seed) < 1./3.) *
        // (!z_r*-STEPSZ + z_r*STEPSZ);

        // Solution 3.
        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
            us = r8_uniform_01(&seed) - 0.5;
            if (us < 0.0)
            dx = -STEPSZ;
            else
            dx = STEPSZ;
        }
        else
            dx = 0.0;

        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
            us = r8_uniform_01(&seed) - 0.5;
            if (us < 0.0)
            dy = -STEPSZ;
            else
            dy = STEPSZ;
        }
        else
            dy = 0.0;

        ut = r8_uniform_01(&seed);
        if (ut < 1.0 / 3.0)
        {
            us = r8_uniform_01(&seed) - 0.5;
            if (us < 0.0)
            dz = -STEPSZ;
            else
            dz = STEPSZ;
        }
        else
            dz = 0.0;


        vs = potentialCuda(x1, x2, x3);
        x1 = x1 + dx;
        x2 = x2 + dy;
        x3 = x3 + dz;

        vh = potentialCuda(x1, x2, x3);

        w = w - 0.5 * H * (vh * ((1.0 - H * vs) * w) + vs * w);

        chk = (x1 / A)*(x1 / A) + (x2 / B)*(x2 / B) + (x3 / C)*(x3 / C);
      }
      sdata[tid] += w;
  }
  __syncthreads();

  // Do reduction in shared memory
  for (int iter = blockDim.x / 2; iter > 0; iter >>= 1) {
    if (tid < iter) {
      sdata[tid] = sdata[tid] + sdata[tid + iter];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) cu_reduction_mem[blockIdx.x] = (w_exact - (sdata[0] / N)) * 
                                                (w_exact - (sdata[0] / N)); // part of err
}


double testCuda(int arc, char **argv)
{
  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  int dim = 3;
  double err;
  double h = 0.001;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int steps_ave;
  double w_exact;
  double wt, chk;


  int N = atoi(argv[1]);
  double time = cpu_time();
  timestamp();

  printf("A = %f\n", a);
  printf("B = %f\n", b);
  printf("C = %f\n", c);
  printf("N = %d\n", N);
  printf("H = %6.4f\n", h);

  stepsz = sqrt((double)dim * h);

  if (a == i4_min(i4_min(a, b), c)){
    ni = 6;
    nj = 1 + i4_ceiling(b / a) * (ni - 1);
    nk = 1 + i4_ceiling(c / a) * (ni - 1);
  }
  else if (b == i4_min(i4_min(a, b), c)){
    nj = 6;
    ni = 1 + i4_ceiling(a / b) * (nj - 1);
    nk = 1 + i4_ceiling(c / b) * (nj - 1);
  }
  else{
    nk = 6;
    ni = 1 + i4_ceiling(a / c) * (nk - 1);
    nj = 1 + i4_ceiling(b / c) * (nk - 1);
  }

  err = 0.0;
  n_inside = 0;
 
  double * points_to_calc = (double *)malloc(4*ni*nj*nk*sizeof(double));
  int ind = 0;
  // Calculate in advance all points that are useful for the calculation on the cpu
  for (int i = 1; i <= ni; i++){
    for (int j = 1; j <= nj; j++){
      for (int k = 1; k <= nk; k++){
        double x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);
        double y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);
        double z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = (x / A)*(x / A) + (y / B)*(y / B) + (z / C)*(z / C);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
            //printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
            //       x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);
          continue;
        }
        n_inside++;
        points_to_calc[ind] = x;
        points_to_calc[ind+1] = y;
        points_to_calc[ind+2] = z;
        points_to_calc[ind+3] = exp((x / a)*(x / a) + (y / b)*(y / b) + (z / c)*(z / c) - 1.0); // w_exact
        ind+=4; // Move ind for the next points
      }
    }
  }
  
  // Get rid off the points that aren't used
  int pointsSize = 4*n_inside*sizeof(double);
  points_to_calc = (double*) realloc(points_to_calc, pointsSize);

  int reduceSize = n_inside*sizeof(double);
  double* reduction_mem = (double*)malloc(reduceSize);

  // Compute number of blocks needed based on array size and desired block size
  int numBlocks = n_inside;
  int sharedMemSize = NUM_OF_THREADS*sizeof(double);
  
  double *cu_points, *cu_reduction_mem;

  cudaMalloc((void **) &cu_points, pointsSize);
  cudaMalloc((void **) &cu_reduction_mem, reduceSize);

  cudaMemcpy(cu_points, points_to_calc, pointsSize, cudaMemcpyHostToDevice);

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(NUM_OF_THREADS);

  feyman_kernel<<< dimGrid, dimBlock, sharedMemSize >>>( cu_points, cu_reduction_mem, N);
  cudaDeviceSynchronize();

  cudaMemcpy(reduction_mem, cu_reduction_mem, reduceSize, cudaMemcpyDeviceToHost);

  // Err reduction on CPU
  for (int i = 0; i < n_inside; ++i) {
    err += reduction_mem[i];
  }
  
  err = sqrt(err / (double)(n_inside));

  // Free memory
  cudaFree(cu_points);
  cudaFree(cu_reduction_mem);
  free(points_to_calc);
  free(reduction_mem);

  time = cpu_time() - time;
  printf("\nRMS absolute error in solution = %e\n", err);
  printf("GPU Time = %f\n", time);
  timestamp();
  printf("\n\n");


  return err;
}

int main(int argc, char **argv) {
    double err_seq, err_cuda;

    err_seq = testSeq(argc, argv);
    err_cuda = testCuda(argc, argv);

    if (abs(err_seq - err_cuda) < ACCURACY) printf("TEST PASSED\n");
    else printf("TEST FAILED\n");
    
    return 0;
}
#include "mpi.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define MASTER 0

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

double potential(double a, double b, double c, double x, double y, double z)
{
  return 2.0 * (pow(x / a / a, 2) + pow(y / b / b, 2) + pow(z / c / c, 2)) + 1.0 / a / a + 1.0 / b / b + 1.0 / c / c;
}

double r8_uniform_01(int *seed)
{
  int k;
  double r;

  k = *seed / 127773;

  *seed = 16807 * (*seed - k * 127773) - k * 2836;

  if (*seed < 0)
  {
    *seed = *seed + 2147483647;
  }
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
int main(int argc, char **argv)
{

  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double a = 3.0;
  double b = 2.0;
  double c = 1.0;
  double chk;
  int dim = 3;
  double dx;
  double dy;
  double dz;
  double err;
  double h = 0.001;
  int i;
  int j;
  int k;
  int n_inside;
  int ni;
  int nj;
  int nk;
  double stepsz;
  int seed = 123456789;
  int steps;
  int steps_ave;
  int trial;
  double us;
  double ut;
  double vh;
  double vs;
  double x;
  double x1;
  double x2;
  double x3;
  double y;
  double w;
  double w_exact;
  double we;
  double wt;
  double z;

  int start_i, start_j, start_k, end_i, end_j, end_k;
  int chunk_i, chunk_j, chunk_k;
  double err_master;
  int N, n_inside_master;

  if (rank == MASTER) {
    N = atoi(argv[1]);
    timestamp();

    printf("A = %f\n", a);
    printf("B = %f\n", b);
    printf("C = %f\n", c);
    printf("N = %d\n", N);
    printf("H = %6.4f\n", h);

    stepsz = sqrt((double)dim * h);    
    if (a == i4_min(i4_min(a, b), c))
    {
        ni = 6;
        nj = 1 + i4_ceiling(b / a) * (ni - 1);
        nk = 1 + i4_ceiling(c / a) * (ni - 1);
    }
    else if (b == i4_min(i4_min(a, b), c))
    {
        nj = 6;
        ni = 1 + i4_ceiling(a / b) * (nj - 1);
        nk = 1 + i4_ceiling(c / b) * (nj - 1);
    }
    else
    {
        nk = 6;
        ni = 1 + i4_ceiling(a / c) * (nk - 1);
        nj = 1 + i4_ceiling(b / c) * (nk - 1);
    }
  }

  MPI_Bcast(&N, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&stepsz, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&ni, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&nj, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  MPI_Bcast(&nk, 1, MPI_INT, MASTER, MPI_COMM_WORLD);
  

  err = 0.0;
  n_inside = 0;

  // Divide outer loops in chunks
  chunk_i = (ni + size - 1) / size;
  chunk_j = (nj + size - 1) / size;
  chunk_k = (nk + size - 1) / size;

  start_i = rank * chunk_i + 1;
  start_j = rank * chunk_j + 1;
  start_k = rank * chunk_k + 1;

  end_i = start_i + chunk_i - 1 <= ni ? start_i + chunk_i - 1 : ni;
  end_j = start_j + chunk_j - 1 <= nj ? start_j + chunk_j - 1 : nj;
  end_k = start_k + chunk_k - 1 <= nk ? start_k + chunk_k - 1 : nk;

  // Make seed different based on rank
  seed = seed - rank;

  for (i = start_i; i <= end_i; i++)
  {
    for (j = start_j; j <= end_j; j++)
    {
      for (k = start_k; k <= end_k; k++)
      {
        x = ((double)(ni - i) * (-a) + (double)(i - 1) * a) / (double)(ni - 1);
        y = ((double)(nj - j) * (-b) + (double)(j - 1) * b) / (double)(nj - 1);
        z = ((double)(nk - k) * (-c) + (double)(k - 1) * c) / (double)(nk - 1);

        chk = pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2);

        if (1.0 < chk)
        {
          w_exact = 1.0;
          wt = 1.0;
          steps_ave = 0;
          // printf("  %7.4f  %7.4f  %7.4f  %10.4e  %10.4e  %10.4e  %8d\n",
          //        x, y, z, wt, w_exact, fabs(w_exact - wt), steps_ave);

          continue;
        }

        n_inside++;

        w_exact = exp(pow(x / a, 2) + pow(y / b, 2) + pow(z / c, 2) - 1.0);

        wt = 0.0;
        steps = 0;
        for (trial = 0; trial < N; trial++)
        {
          x1 = x;
          x2 = y;
          x3 = z;
          w = 1.0;
          chk = 0.0;
          while (chk < 1.0)
          {
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

  MPI_Reduce(&err, &err_master, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);
  MPI_Reduce(&n_inside, &n_inside_master, 1, MPI_INT, MPI_SUM, MASTER, MPI_COMM_WORLD);

  if (rank == MASTER) {
    err = sqrt(err_master / (double)(n_inside_master));

    printf("\n\nRMS absolute error in solution = %e\n", err);
    timestamp();
  }

  MPI_Finalize();
  return 0;
}

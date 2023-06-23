#include <stdio.h>
#include <math.h>
#include <time.h>

#define mm 15
#define npart 4 * mm *mm *mm
#define NUM_OF_GPU_THREADS 1024

/*
 *  Function declarations
 */

void dfill(int, double, double[], int);

void domove(int, double[], double[], double[], double);

void dscal(int, double, double[], int);

void fcc(double[], int, int, double);

void forces(int, double[], double[], double, double);
__global__ void forcesGPU(int, double, double, double*, double*, double*);

double mkekin(int, double[], double[], double, double);

void mxwell(double[], int, double, double);

void prnout(int, double, double, double, double, double, double, int, double);

double velavg(int, double[], double, double);

double secnds(void);

/*
 *  Variable declarations
 */

double epot;
double vir;
double count;



double cpu_time(void)
{
  double value;

  value = (double)clock() / (double)CLOCKS_PER_SEC;

  return value;
}
/*
 *  Main program : Molecular Dynamics simulation.
 */
int testCPU()
{
  int move;
  double x[npart * 3], vh[npart * 3], f[npart * 3];
  double ekin;
  double vel;
  double sc;
  double start, time;

  /*
   *  Parameter definitions
   */

  double den = 0.83134;
  double side = pow((double)npart / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npart - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  /*
   *  Initial output
   */

  printf(" Molecular Dynamics Simulation example program\n");
  printf(" ---------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npart);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  /*
   *  Generate fcc lattice for atoms inside box
   */
  fcc(x, npart, mm, a);
  /*
   *  Initialise velocities and forces (which are zero in fcc positions)
   */
  mxwell(vh, 3 * npart, h, tref);
  dfill(3 * npart, 0.0, f, 1);
  /*
   *  Start of md
   */
  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = cpu_time();

  for (move = 1; move <= movemx; move++)
  {

    /*
     *  Move the particles and partially update velocities
     */
    domove(3 * npart, x, vh, f, side);

    /*
     *  Compute forces in the new positions and accumulate the virial
     *  and potential energy.
     */
    
    forces(npart, x, f, side, rcoff);
    //printf("vir: %f, ", vir);
    /*
     *  Scale forces, complete update of velocities and compute k.e.
     */
    ekin = mkekin(npart, f, vh, hsq2, hsq);

    /*
     *  Average the velocity and temperature scale if desired
     */
    vel = velavg(npart, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npart, sc, vh, 1);
      ekin = tref / tscale;
    }
    
    /*
     *  Sum to get full potential energy and virial
     */
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
  }

  time = cpu_time() - start;

  printf("Time =  %f\n", (float)time);
}

int testGPU()
{
  int move;
  double x[npart * 3], vh[npart * 3], f[npart * 3];
  double ekin;
  double vel;
  double sc;
  double start, time;
  double *rez;
  /*
   *  Parameter definitions
   */

  double den = 0.83134;
  double side = pow((double)npart / den, 0.3333333);
  double tref = 0.722;
  double rcoff = (double)mm / 4.0;
  double h = 0.064;
  int irep = 10;
  int istop = 20;
  int iprint = 5;
  int movemx = 20;

  double a = side / (double)mm;
  double hsq = h * h;
  double hsq2 = hsq * 0.5;
  double tscale = 16.0 / ((double)npart - 1.0);
  double vaver = 1.13 * sqrt(tref / 24.0);

  const int numThreadsPerBlock = NUM_OF_GPU_THREADS;
  int numBlocks = ceil(1.0 * npart / numThreadsPerBlock);
  double sharedMemSize = (numThreadsPerBlock * 2 + 100)* sizeof(double);
  double* d_f, *d_x,*d_rez;

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);

  rez = (double*)malloc(2*numBlocks*sizeof(double));
  

  cudaMalloc(&d_f, 3*npart*sizeof(double));
  cudaMalloc(&d_x, 3*npart*sizeof(double));
  cudaMalloc(&d_rez, 2*numBlocks*sizeof(double));
  /*
   *  Initial output
   */

  printf(" Molecular Dynamics Simulation example program\n");
  printf(" ---------------------------------------------\n");
  printf(" number of particles is ............ %6d\n", npart);
  printf(" side length of the box is ......... %13.6f\n", side);
  printf(" cut off is ........................ %13.6f\n", rcoff);
  printf(" reduced temperature is ............ %13.6f\n", tref);
  printf(" basic timestep is ................. %13.6f\n", h);
  printf(" temperature scale interval ........ %6d\n", irep);
  printf(" stop scaling at move .............. %6d\n", istop);
  printf(" print interval .................... %6d\n", iprint);
  printf(" total no. of steps ................ %6d\n", movemx);

  /*
   *  Generate fcc lattice for atoms inside box
   */
  fcc(x, npart, mm, a);
  /*
   *  Initialise velocities and forces (which are zero in fcc positions)
   */
  mxwell(vh, 3 * npart, h, tref);
  dfill(3 * npart, 0.0, f, 1);
  /*
   *  Start of md
   */
  printf("\n    i       ke         pe            e         temp   "
         "   pres      vel      rp\n  -----  ----------  ----------"
         "  ----------  --------  --------  --------  ----\n");

  start = cpu_time();

  for (move = 1; move <= movemx; move++)
  {

    /*
     *  Move the particles and partially update velocities
     */
    domove(3 * npart, x, vh, f, side);

    /*
     *  Compute forces in the new positions and accumulate the virial
     *  and potential energy.
     */
    
    cudaMemcpy(d_x, x, 3*npart*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, 3*npart*sizeof(double), cudaMemcpyHostToDevice);
    
    forcesGPU<<< dimGrid, dimBlock, sharedMemSize >>>(npart, side, rcoff, d_x, d_f, d_rez);
    
    cudaMemcpy(f, d_f, 3*npart*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(rez, d_rez, 2*numBlocks*sizeof(double), cudaMemcpyDeviceToHost);
    vir = 0.0;
    epot = 0.0;
    for (int i = 0; i < 2*numBlocks; i+=2) {
      epot += rez[i];
      vir -= rez[i+1];
    }
    //printf("vir: %f, ", vir);
    /*
     *  Scale forces, complete update of velocities and compute k.e.
     */
    ekin = mkekin(npart, f, vh, hsq2, hsq);

    /*
     *  Average the velocity and temperature scale if desired
     */
    vel = velavg(npart, vh, vaver, h);
    if (move < istop && fmod(move, irep) == 0)
    {
      sc = sqrt(tref / (tscale * ekin));
      dscal(3 * npart, sc, vh, 1);
      ekin = tref / tscale;
    }

    /*
     *  Sum to get full potential energy and virial
     */
    if (fmod(move, iprint) == 0)
      prnout(move, ekin, epot, tscale, vir, vel, count, npart, den);
  }
  cudaFree(d_f);
  cudaFree(d_x);

  time = cpu_time() - start;

  printf("Time =  %f\n", (float)time);
}

int main(){
  testCPU();
  epot = 0.0;
  vir = 0.0;
  count = 0.0;
  cudaDeviceSynchronize();
  testGPU();
}

time_t starttime = 0;


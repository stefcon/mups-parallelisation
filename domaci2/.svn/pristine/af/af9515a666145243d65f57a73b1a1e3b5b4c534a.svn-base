#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <string.h>

#define mm 15
#define npart 4 * mm *mm *mm
/*
 *  Function declarations
 */

void dfill(int, double, double[], int);

void domove(int, double[], double[], double[], double);

void dscal(int, double, double[], int);

void fcc(double[], int, int, double);

void forces(int, double[], double[], double, double, int, int);

double
mkekin(int, double[], double[], double, double);

void mxwell(double[], int, double, double);

void prnout(int, double, double, double, double, double, double, int, double);

double
velavg(int, double[], double, double);

double
secnds(void);

/*
 *  Variable declarations
 */

double epot;
double vir;
double count;

/*
 *  Main program : Molecular Dynamics simulation.
 */

#define TAG_WAIT 0
#define TAG_ASK_FOR_JOB 1
#define TAG_JOB_DATA 2
#define TAG_STOP 3
#define TAG_RESULT 4

int main(int argc, char *argv[])
{
  int rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int chunk = 1500;
  MPI_Status stat, stat2;


  int move;
  double x[npart * 3], vh[npart * 3], f[npart * 3], f_manager[npart*3];
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
  if(rank == 0){
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

    start = secnds();
  

    int cur = 0;
    int size;
    int done = 0;
    int buf = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int working = 0;
    int not_stopped = ~(1 << size-1);

    for (move = 1; move <= movemx; move++)
    {

      epot = vir = count = 0;
      domove(3*npart, x, vh, f, side);
    
      do{
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
        int slave_rank = stat.MPI_SOURCE;
        // Decide according to the tag which type of message we have got
        if (stat.MPI_TAG == TAG_ASK_FOR_JOB) {
          int rank;
          MPI_Recv(&rank, 1, MPI_INT, slave_rank, TAG_ASK_FOR_JOB, MPI_COMM_WORLD, &stat2);
          if (cur < npart*3) {
            working++;

            MPI_Send(&cur, 1, MPI_INT, slave_rank, TAG_JOB_DATA, MPI_COMM_WORLD);
            MPI_Send(x, 3*npart, MPI_DOUBLE, slave_rank, TAG_JOB_DATA, MPI_COMM_WORLD);

            /* mark slave with rank my_rank as working on a job */
            cur += chunk;
          } else if(move==movemx){
            // send stop msg to slave
            not_stopped &= ~(1 << (slave_rank - 1));
            MPI_Send (&buf, 1, MPI_INT, slave_rank, TAG_STOP, MPI_COMM_WORLD);
          } else{
            MPI_Send (&buf, 1, MPI_INT, slave_rank, TAG_WAIT, MPI_COMM_WORLD);
          }
        } else {
          working--;
          
          int indx;
          double epot_m, vir_m;
          MPI_Recv(&indx, 1, MPI_INT, slave_rank, TAG_RESULT, MPI_COMM_WORLD, &stat2);
          MPI_Recv(&epot_m, 1, MPI_DOUBLE, slave_rank, TAG_RESULT, MPI_COMM_WORLD, &stat2);
          MPI_Recv(&vir_m, 1, MPI_DOUBLE, slave_rank, TAG_RESULT, MPI_COMM_WORLD, &stat2);
          MPI_Recv(f_manager, 3 * npart, MPI_DOUBLE, slave_rank, TAG_RESULT, MPI_COMM_WORLD, &stat2);
          // Reduce variables
          for (int i = indx; i < npart*3; ++i) {
            f[i] += f_manager[i];
          }
          epot += epot_m;
          vir += vir_m;
        }
        
      } while (cur < npart*3 || working > 0);
      // Reset cur for next iteration
      cur = 0;

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
    // Tell everyone that the job is done
    for (int i = 1; i < size; ++i) {
      if (not_stopped & (1 << i-1)) {
        MPI_Send(&buf, 1, MPI_INT, i, TAG_STOP, MPI_COMM_WORLD);
      }
    }

    time = secnds() - start;

    printf("Time =  %f\n", (float)time);
  }
  else {
    int stopped = 0;
    MPI_Status stat , stat2 ;
    do {
      MPI_Send(&rank, 1, MPI_INT, 0, TAG_ASK_FOR_JOB, MPI_COMM_WORLD);
      int buf;
      
      MPI_Probe (0 ,MPI_ANY_TAG, MPI_COMM_WORLD , &stat);
      
      if (stat.MPI_TAG == TAG_JOB_DATA) {
          int start;
          MPI_Recv(&start , 1, MPI_INT, 0, TAG_JOB_DATA, MPI_COMM_WORLD, &stat2);
          MPI_Recv(x, 3*npart, MPI_DOUBLE, 0, TAG_JOB_DATA, MPI_COMM_WORLD, &stat2);

          // Initilise f with zeros
          memset(f, 0, sizeof(double)*(npart*3));

          /*
          *  Compute forces in the new positions and accumulate the virial
          *  and potential energy.
          */
          forces(npart, x, f, side, rcoff, start, chunk);

          MPI_Send(&start, 1, MPI_INT, 0, TAG_RESULT, MPI_COMM_WORLD);
          MPI_Send(&epot, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
          MPI_Send(&vir, 1, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
          MPI_Send(f, 3 * npart, MPI_DOUBLE, 0, TAG_RESULT, MPI_COMM_WORLD);
          
          //printf("rank: %d, done: %d\n", rank,  start);
      } else if(stat.MPI_TAG == TAG_STOP) {
          // We got a stop message we have to retrieve it by using MPI_Recv
          // But we can ignore the data from the MPI_Recv call
          
          MPI_Recv (&buf, 1, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &stat2);
          stopped = 1;
      }else{
        MPI_Recv(&buf, 1, MPI_INT, 0, TAG_WAIT, MPI_COMM_WORLD, &stat2);
      }
    } while (stopped == 0);
  }

  MPI_Finalize();
}

time_t starttime = 0;

double secnds()
{
  return omp_get_wtime();
}

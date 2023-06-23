__global__ void forcesGPU(int npart, double side, double rcoff, double *d_x, double *d_f, double *d_rez){
  __shared__ double ekin[1024];
  __shared__ double vir[1024];
  double xi, yi, zi, fxi, fyi, fzi, xx, yy, zz;
  double rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
  double forcex, forcey, forcez;
  int j;
  double sideh = 0.5*side;

  int tid = threadIdx.x;
  int i = (blockIdx.x * blockDim.x + tid)*3;
  ekin[tid] = 0.0;
  vir[tid] = 0.0;

  xi = d_x[i];
  yi = d_x[i + 1];
  zi = d_x[i + 2];
  fxi = 0.0;
  fyi = 0.0;
  fzi = 0.0;

  for (j = i + 3; j < npart * 3; j += 3){
    xx = xi - d_x[j];
    yy = yi - d_x[j + 1];
    zz = zi - d_x[j + 2];
    if (xx < -sideh)
      xx += side;
    if (xx > sideh)
      xx -= side;
    if (yy < -sideh)
      yy += side;
    if (yy > sideh)
      yy -= side;
    if (zz < -sideh)
      zz += side;
    if (zz > sideh)
      zz -= side;
    rd = xx * xx + yy * yy + zz * zz;

    if (rd <= rcoff*rcoff)
    {
      rrd = 1.0 / rd;
      rrd2 = rrd * rrd;
      rrd3 = rrd2 * rrd;
      rrd4 = rrd2 * rrd2;
      rrd6 = rrd2 * rrd4;
      rrd7 = rrd6 * rrd;
      ekin[tid] += (rrd6 - rrd3);
      r148 = rrd7 - 0.5 * rrd4;
      vir[tid] += rd * r148;
      forcex = xx * r148;
      fxi += forcex;
      forcey = yy * r148;
      fyi += forcey;
      forcez = zz * r148;
      fzi += forcez;
      
      atomicAdd(&d_f[j], -1*forcex);
      atomicAdd(&d_f[j + 1], -1*forcey);
      atomicAdd(&d_f[j + 2], -1*forcez);
    }
  }


  d_f[i] += fxi;
  d_f[i + 1] += fyi;
  d_f[i + 2] += fzi;

  __syncthreads();

  for (int iter = blockDim.x/2; iter > 0; iter >>= 1) {
    if (tid < iter) {
      ekin[tid] = ekin[tid] + ekin[tid + iter];
      vir[tid] = vir[tid] + vir[tid + iter];
    }
  }
  __syncthreads();
  
  if (tid == 0) d_rez[2*blockIdx.x] = ekin[0];
  if (tid == 0) d_rez[2*blockIdx.x + 1] = vir[0];

}


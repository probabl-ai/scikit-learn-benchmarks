// Cost of changing the OpenMP team size between consecutive parallel regions.
// usage: team BIG SMALL N_ITER ITEM_US
// mode const: every region uses BIG threads; mode alt: alternate BIG and SMALL.
#define _GNU_SOURCE
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <time.h>
static double now_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static void busy(double us){double t0=now_us();while(now_us()-t0<us);}
#define MAXTID 1000000
static char seen[MAXTID];
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  int big=atoi(argv[1]),small=atoi(argv[2]),n=atoi(argv[3]);double item=atof(argv[4]);
  double *tb=malloc(n*sizeof(double)),*ts=malloc(n*sizeof(double));
  for(int mode=0;mode<2;mode++){
    for(int i=0;i<MAXTID;i++)seen[i]=0;
    int nb=0,ns=0,ntid=0;
    for(int it=0;it<n;it++){
      int k=(mode==1&&(it&1))?small:big;
      double t0=now_us();
      #pragma omp parallel num_threads(k)
      { int tid=syscall(SYS_gettid)%MAXTID; if(!seen[tid]){seen[tid]=1;
          #pragma omp atomic
          ntid++;}
        busy(item); }
      double dt=now_us()-t0;
      if(k==big)tb[nb++]=dt;else ts[ns++]=dt;
    }
    qsort(tb,nb,sizeof(double),cmp);
    printf("%s big=%d small=%d  region_us(big) med=%.1f p90=%.1f",mode?"alt  ":"const",big,small,tb[nb/2],tb[nb*9/10]);
    if(ns){qsort(ts,ns,sizeof(double),cmp);printf("  region_us(small) med=%.1f",ts[ns/2]);}
    printf("  distinct_tids=%d\n",ntid);
  }
  return 0;
}

// Two parallel regions vs one region with a barrier between two loops.
// usage: barrier ITEMS_PER_THREAD ITEM_US OUTER_US N_ITER GAP_US...
//   2reg     : parallel for; serial gap GAP_US; parallel for            (what two prange give)
//   1reg+bar : parallel { for; [implicit barrier] for }                 (fused, dependent loops)
//   1reg     : parallel { for over both phases' items }                 (fused, independent)
// Each iteration ends with OUTER_US of serial work (rest of the grower loop).
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now_us(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec/1e3;}
static void busy(double us){double t0=now_us();while(now_us()-t0<us);}
static int cmp(const void*a,const void*b){double x=*(double*)a,y=*(double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
  int k=atoi(argv[1]);double w=atof(argv[2]),outer=atof(argv[3]);int n=atoi(argv[4]);
  int T=omp_get_max_threads(),m=k*T;double*ts=malloc(n*sizeof(double));
  const char*names[]={"2reg","1reg+bar","1reg"};
  for(int g=5;g<argc;g++){double gap=atof(argv[g]);
   for(int mode=0;mode<3;mode++){ if(mode>0&&g>5)continue;  // gap only matters for 2reg
    for(int it=0;it<n;it++){
      double t0=now_us();
      if(mode==0){
        #pragma omp parallel for schedule(static)
        for(int i=0;i<m;i++)busy(w);
        busy(gap);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<m;i++)busy(w);
      }else if(mode==1){
        #pragma omp parallel
        {
          #pragma omp for schedule(static)
          for(int i=0;i<m;i++)busy(w);
          #pragma omp for schedule(static) nowait
          for(int i=0;i<m;i++)busy(w);
        }
      }else{
        #pragma omp parallel for schedule(static)
        for(int i=0;i<2*m;i++)busy(w);
      }
      double dt=now_us()-t0-(mode==0?gap:0);
      ts[it]=dt; busy(outer);
    }
    qsort(ts,n,sizeof(double),cmp);
    double ideal=2*k*w;
    printf("T=%d k=%d item=%5.1fus outer=%4.0fus %-9s gap=%4.0fus  med=%7.1f p90=%7.1f  overhead_med=%6.1fus\n",
      T,k,w,outer,names[mode],mode==0?gap:0,ts[n/2],ts[n*9/10],ts[n/2]-ideal);
   }
  }
  return 0;
}

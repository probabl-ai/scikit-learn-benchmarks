/* Which clause combinations on `omp for nowait` emit a barrier? */
void work(int);
void firstonly(int n){int k;int i=0;
#pragma omp parallel
{
#pragma omp for nowait firstprivate(i) schedule(static)
for(k=0;k<n;k++){work(i+k);}
}
}
void both(int n, int *out){int k;int i=0;
#pragma omp parallel
{
#pragma omp for nowait firstprivate(i) lastprivate(i) schedule(static)
for(k=0;k<n;k++){i=k;work(i);}
}
*out=i;}
void lastonly(int n, int *out){int k;int i=0;
#pragma omp parallel
{
#pragma omp for nowait lastprivate(i) schedule(static)
for(k=0;k<n;k++){i=k;work(i);}
}
*out=i;}
void neither(int n){int k;
#pragma omp parallel
{
#pragma omp for nowait schedule(static)
for(k=0;k<n;k++){work(k);}
}
}

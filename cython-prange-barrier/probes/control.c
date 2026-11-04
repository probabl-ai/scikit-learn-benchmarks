/* Positive control: does GCC warn on OpenMP privates at all? Each function reads a possibly uninitialized value. */
void work(int);
void plain_maybe(int n, int *out){int x; for(int k=0;k<n;k++){x=k;} *out=x;}
void read_before_write(int n, int *out){int x=0;
#pragma omp parallel
{
#pragma omp for nowait lastprivate(x) schedule(static)
for(int k=0;k<n;k++){work(x);x=k;}
}
*out=x;}
void conditional(int n, int *a, int *out){int x=0;
#pragma omp parallel
{
#pragma omp for nowait lastprivate(x) schedule(static)
for(int k=0;k<n;k++){if(a[k])x=k;}
}
*out=x;}
void private_read(int n){int x;
#pragma omp parallel
{
#pragma omp for nowait private(x) schedule(static)
for(int k=0;k<n;k++){work(x);x=k;}
}
}

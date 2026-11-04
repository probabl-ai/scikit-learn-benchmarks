#define BAR() __pragma(omp barrier)
void work(int);
void f(int n){
#pragma omp parallel
{ work(1);
  BAR();
  work(2);
}
}

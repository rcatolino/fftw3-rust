#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

clock_t sclock() {
  // The clock() call only measures the cpu time. (CLOCK_PROCESS_CPUTIME_ID)
  // This measures the total time spent.
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000000000 + ts.tv_nsec;
}

void bench_complex(unsigned int N) {
  clock_t time_ini, time_exec = 0;
  fftw_complex *in,*out;
  fftw_complex input[N];

  for (unsigned int i = 0; i<N; i++) {
    input[0][i] = 200*(drand48() - 0.5);
    input[1][i] = 200*(drand48() - 0.5);
  }

  for (int i = 0; i<1000; i++) {
    clock_t start, t1, t2;
    fftw_plan p;

    start = sclock();
    in = fftw_alloc_complex(N);
    memcpy(in, input, N*sizeof(double[2]));
    out = fftw_alloc_complex(N);
    p = fftw_plan_dft_1d(N, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    t1 = sclock();

    fftw_execute(p);
    t2 = sclock();

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    time_ini = (i*time_ini + (t1-start))/(i+1);
    time_exec = (i*time_exec + (t2-t1))/(i+1);
  }

  printf("n=%u, complex, init: %fms, exec: %fms\n", N, time_ini/1000.0, time_exec/1000.0);
}
void bench_real(unsigned int N) {
  clock_t time_ini, time_exec = 0;
  fftw_complex *out;
  double input[N];

  for (unsigned int i = 0; i<N; i++) {
    input[i] = 200*(drand48() - 0.5);
  }

  for (int i = 0; i<1000; i++) {
    clock_t start, t1, t2;
    double *in;
    fftw_plan p;

    start = sclock();
    in = fftw_alloc_real(N);
    memcpy(in, input, N*sizeof(double));
    out = fftw_alloc_complex(N/2 + 1);
    p = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    t1 = sclock();

    fftw_execute(p);
    t2 = sclock();

    fftw_destroy_plan(p);
    fftw_free(in); fftw_free(out);
    time_ini = (i*time_ini + (t1-start))/(i+1);
    time_exec = (i*time_exec + (t2-t1))/(i+1);
  }
  printf("n=%u, real, init: %fms, exec: %fms\n", N, time_ini/1000.0, time_exec/1000.0);
}

int main() {
  bench_complex(100);
  bench_complex(10000);
  bench_real(100);
  bench_real(10000);
  return 0;
}


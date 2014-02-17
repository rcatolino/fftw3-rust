// Copyright (c) 2014 Raphael Catolino
#[cfg(test)];

use extra::time::precise_time_ns;
use num::complex::Cmplx;

use super::{Fftw, TransformInput};

mod fftw3_macros;

#[test]
fn test_1d_cmplx() {
  let inp = ca!{48 -2, 39 + 5, 37+3, 55+0, 22+210};
  let mut fftw = Fftw::from_slice(inp);
  fftw.compute().unwrap();
}

#[test]
fn test_1d_from_slice() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1, -3};
  let mut fftw = Fftw::from_slice(inp);
  fftw.compute().unwrap();
}

#[test]
fn test_1d() {
  let mut fftw = Fftw::new(7);
  for i in range(-2f64, 5f64) {
    fftw.ref_input().push(i);
  }
  fftw.compute().unwrap();
}

#[test]
fn test_1d_overflow() {
  let mut fftw = Fftw::new(7);
  for i in range(-2f64, 5f64) {
    fftw.ref_input().push(i);
  }
  fftw.ref_input().push(42f64) && fail!();
  fftw.compute().unwrap();
}

#[test]
fn test_1d_uncomplete() {
  let mut fftw = Fftw::new(4);
  {
    let inp = fftw.ref_input();
    inp.push(3f64);
    inp.push(2f64);
    inp.push(-3f64);
  }
  fftw.compute().is_some() && fail!();
}

#[test]
fn test_iter_few() {
  let inp = [~[], hra!{1}, hra!{1, -5}, hra!{1, -2, -5}, hra!{-2, 46, 2, 1}];
  for inn in inp.iter() {
    let mut fftw = Fftw::from_slice(inn.as_slice());
    fftw.compute();
    let it = fftw.iter_symmetry();
    for (i,j) in it.zip(fftw.output().iter()) {
      assert!(i == *j);
    }
    for (i,j) in it.skip(fftw.output().len())
      .zip(fftw.output().rev_iter().skip((fftw.input().len()+1) % 2)) {
      assert!(i == j.conj());
    }
  }
}

#[test]
fn test_iter_odd() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1, -3};
  let mut fftw = Fftw::from_slice(inp);
  fftw.compute();
  let it = fftw.iter_symmetry();
  for (i,j) in it.zip(fftw.output().iter()) {
    assert!(i == *j);
  }
  for (i,j) in it.skip(fftw.output().len()).zip(fftw.output().rev_iter()) {
    assert!(i == j.conj());
  }
}

#[test]
fn test_iter_even() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1};
  let mut fftw = Fftw::from_slice(inp);
  fftw.compute();
  let it = fftw.iter_symmetry();
  for (i,j) in it.zip(fftw.output().iter()) {
    assert!(i == *j);
  }
  for (i,j) in it.skip(fftw.output().len()).zip(fftw.output().rev_iter().skip(1)) {
    assert!(i == j.conj());
  }
}

#[test]
fn test_index() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1};
  let fftw = Fftw::from_slice(inp);
  for i in range(0, 8) {
    assert!(inp[i] == fftw.input()[i as uint]);
  }

  fftw.input().get(8).is_none() || fail!();
}

fn bench<T: TransformInput>(slice: &[T], domain: &str) {
  let (mut t1, mut t2) = (0u64, 0u64);
  for i in range(0u64, 1000u64) {
    let start = precise_time_ns();
    let mut fftw = Fftw::from_slice(slice);
    let time1 = precise_time_ns();
    fftw.compute();
    let time2 = precise_time_ns();
    t1 = (i*t1 + (time1-start))/(i+1);
    t2 = (i*t2 + (time2-time1))/(i+1);
  }

  let _ = ::std::io::stdout().write_line(format!("n={}, {}, time elapsed : {}ms, {}ms",
                                                 slice.len(), domain, t1 as f64/1000f64,
                                                 t2 as f64/1000f64));
}

#[test]
fn bench_real_10000() {
  use std::rand::{rng, Rng};

  let mut rng = rng();
  let mut buff = ::std::vec::with_capacity(10000);
  for _ in range(0, 10000) {
    buff.push(rng.gen_range(-100f64, 100f64));
  }

  bench(buff.as_slice(), "real");
}

#[test]
fn bench_real_100() {
  use std::rand::{rng, Rng};

  let mut rng = rng();
  let mut buff = ::std::vec::with_capacity(100);
  for _ in range(0, 100) {
    buff.push(rng.gen_range(-100f64, 100f64));
  }

  bench(buff.as_slice(), "real");
}

#[test]
fn bench_cmplx_10000() {
  use std::rand::{rng, Rng};

  let mut rng = rng();
  let mut buff = ::std::vec::with_capacity(10000);
  for _ in range(0, 10000) {
    buff.push(Cmplx::new(rng.gen_range(-100f64, 100f64),
                         rng.gen_range(-100f64, 100f64)));
  }

  bench(buff.as_slice(), "complex");
}

#[test]
fn bench_cmplx_100() {
  use std::rand::{rng, Rng};

  let mut rng = rng();
  let mut buff = ::std::vec::with_capacity(100);
  for _ in range(0, 100) {
    buff.push(Cmplx::new(rng.gen_range(-100f64, 100f64),
                         rng.gen_range(-100f64, 100f64)));
  }

  bench(buff.as_slice(), "complex");
}

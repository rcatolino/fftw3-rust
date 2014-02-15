// Copyright (c) 2014 Raphael Catolino
#[cfg(test)];

use extra::complex::Cmplx;

use super::{Fftw};

mod fftw3_macros;

#[test]
fn test_1d_cmplx() {
  let inp = ca!{48 -2, 39 + 5, 37+3, 55+0, 22+210};
  let mut fftw = Fftw::from_slice(inp);
  fftw.fft().unwrap();
}

#[test]
fn test_1d_from_slice() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1, -3};
  let mut fftw = Fftw::from_slice(inp);
  fftw.fft().unwrap();
}

#[test]
fn test_1d() {
  let mut fftw = Fftw::new(7);
  for i in range(-2f64, 5f64) {
    fftw.mut_input().push(i);
  }
  fftw.fft().unwrap();
}

#[test]
fn test_1d_overflow() {
  let mut fftw = Fftw::new(7);
  for i in range(-2f64, 5f64) {
    fftw.mut_input().push(i);
  }
  fftw.mut_input().push(42f64) && fail!();
  fftw.fft().unwrap();
}

#[test]
fn test_1d_uncomplete() {
  let mut fftw = Fftw::new(4);
  {
    let inp = fftw.mut_input();
    inp.push(3f64);
    inp.push(2f64);
    inp.push(-3f64);
  }
  fftw.fft().is_some() && fail!();
}

#[test]
fn test_iter_few() {
  let inp = [~[], hra!{1}, hra!{1, -5}, hra!{1, -2, -5}, hra!{-2, 46, 2, 1}];
  for inn in inp.iter() {
    let mut fftw = Fftw::from_slice(inn.as_slice());
    fftw.fft();
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
  fftw.fft();
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
  fftw.fft();
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
    assert!(inp[i] == fftw.input()[i as uint].unwrap());
  }

  fftw.input()[8].is_none() || fail!();
}

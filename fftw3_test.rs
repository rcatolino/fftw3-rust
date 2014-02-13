#[cfg(test)];


use extra::complex::Cmplx;
use super::{Fftw};

macro_rules! hra {
  ($($item:expr),+) => (
    ~[$($item as f64),+]
  )
}

macro_rules! ra {
  ($($item:expr),+) => (
    [$($item as f64),+]
  )
}

macro_rules! ca {
  ($($re:tt $sign:tt $im:tt),+) => (
    [$(_c!{$re, $sign $im}),+]
  )
}

macro_rules! _c {
  ($re:expr , + $im:expr) => {
    Cmplx::new($re as f64, $im as f64)
  };
  ($re:expr , - $im:expr) => {
    Cmplx::new($re as f64, -$im as f64)
  }
}

macro_rules! c {
  ($re:expr , $im:expr) => {
    Cmplx::new($re as f64, $im as f64)
  };
  ($re:expr) => {
    Cmplx::new($re as f64, 0f64)
  }
}

#[test]
fn test_1d_cmplx() {
  let inp = ca!{48 -2, 39 + 5, 37+3, 55+0, 22+210};
  let mut fftw = Fftw::from_slice(inp);
  fftw.fft().unwrap();
}

#[test]
fn test_1d_real_from_slice() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1, -3};
  let mut fftw = Fftw::from_slice_real(inp);
  fftw.fft().unwrap();
}

#[test]
fn test_1d_real() {
  let mut fftw = Fftw::new_real(7);
  for i in range(-2f64, 5f64) {
    fftw.push(i);
  }
  fftw.fft().unwrap();
}

#[test]
fn test_1d_real_overflow() {
  let mut fftw = Fftw::new_real(7);
  for i in range(-2f64, 5f64) {
    fftw.push(i);
  }
  fftw.push(42f64) && fail!();
  fftw.fft().unwrap();
}

#[test]
fn test_1d_real_uncomplete() {
  let mut fftw = Fftw::new_real(4);
  fftw.push(3f64);
  fftw.push(2f64);
  fftw.push(-3f64);
  fftw.fft().is_some() && fail!();
}

#[test]
fn test_real_iter_few() {
  let inp = [~[], hra!{1}, hra!{1, -5}, hra!{1, -2, -5}, hra!{-2, 46, 2, 1}];
  for inn in inp.iter() {
    let mut fftw = Fftw::from_slice_real(inn.as_slice());
    fftw.fft();
    let it = fftw.iter();
    for (i,j) in it.zip(fftw.result().iter()) {
      assert!(i == *j);
    }
    for (i,j) in it.skip(fftw.result().len())
      .zip(fftw.result().rev_iter().skip((fftw.len()+1) % 2)) {
      assert!(i == j.conj());
    }
  }
}

#[test]
fn test_real_iter_odd() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1, -3};
  let mut fftw = Fftw::from_slice_real(inp);
  fftw.fft();
  let it = fftw.iter();
  for (i,j) in it.zip(fftw.result().iter()) {
    assert!(i == *j);
  }
  for (i,j) in it.skip(fftw.result().len()).zip(fftw.result().rev_iter()) {
    assert!(i == j.conj());
  }
}

#[test]
fn test_real_iter_even() {
  let inp = ra!{1, 0, 2, 4, 5, 2, 0, -1};
  let mut fftw = Fftw::from_slice_real(inp);
  fftw.fft();
  let it = fftw.iter();
  for (i,j) in it.zip(fftw.result().iter()) {
    assert!(i == *j);
  }
  for (i,j) in it.skip(fftw.result().len()).zip(fftw.result().rev_iter().skip(1)) {
    assert!(i == j.conj());
  }
}

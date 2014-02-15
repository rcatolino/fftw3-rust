// Copyright (c) 2014 Raphael Catolino
#[feature(macro_rules)];
#[crate_id = "fftw3_rust"];
#[crate_type = "lib"];
#[crate_type = "dylib"];

#[cfg(target_arch = "x86_64")]
extern mod extra;
extern mod sync;

use extra::complex::Cmplx;

use fftw3_bindgen::{FFTW_FORWARD, FFTW_ESTIMATE, fftw_alloc_complex, fftw_free,
                    fftw_plan_dft_1d, fftw_plan_dft_r2c_1d, fftw_destroy_plan,
                    fftw_alloc_real, fftw_execute, fftw_plan};

use std::cast::transmute;
use std::fmt::{Show, Formatter};
use std::fmt;
use std::libc::{c_int, c_void, size_t};
use std::mem::move_val_init;
use std::ptr::copy_memory;

use sync::mutex::{StaticMutex, MUTEX_INIT};
pub use iteration::HermitianItems;

mod fftw3_bindgen;
mod fftw3_test;
pub mod fftw3_macros;

static mut LOCK: StaticMutex = MUTEX_INIT;

/** Pretty-print an array of complex :

  ```
  let cmplx_array = ca!{48 -2, 39 + 5, 37+3, 55+0, 22+210};
  println!("{}", Line(cmplx_array));
  println!("{}", Col(cmplx_array));
  ```
**/

pub enum CxDisplay<'a> {
  Line(&'a[Cmplx<f64>]),
  Col(&'a[Cmplx<f64>]),
}

impl<'a> Show for CxDisplay<'a> {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let (sep, array) = match self {
      &Line(ref array) => (' ', array),
      &Col(ref array) => ('\n', array),
    };

    if array.is_empty() {
      write!(f.buf, "[ ]")
    } else {
      if_ok!(write!(f.buf, "[{}{:f} {:+f}i", sep, array[0].re, array[0].im));
      for cx in array.iter().skip(1) {
        if_ok!(write!(f.buf, ",{}{:f} {:+f}i", sep, cx.re, cx.im));
      }
      write!(f.buf, "{}]", sep)
    }
  }
}

priv struct CplxSlice<T> {
  data: *T,
  len: uint,
}

/** Holds the state of a fourier transform :

    Compute the 1d transform for real values :

    ```rust
    use fftw3_rust::{Fftw, Line};

    let input = [1f64, 0f64, 2f64, 4f64, 5f64, 2f64, 0f64, -1f64, -3f64];
    let mut fftw = Fftw::from_slice(input);
    fftw.fft();
    println!("{}", Line(fftw.output()));
    ```

    Compute the 1d transform for complex values :

    ```rust
    use extra::complex::Cmplx;
    use fftw3_rust::{Fftw, Line};

    let input = [Cmplx::new(1f64, -1f64), Cmplx::new(0f64, 2f64), Cmplx::new(4f64, 12f64)];
    let mut fftw = Fftw::from_slice(input);
    fftw.fft();
    println!("{}", Line(fftw.output()));
    ```
**/
pub struct Fftw<T> {
  priv time_dom: FftBuf<T>,
  priv freq_dom: FftBuf<Cmplx<f64>>,
  priv plan: fftw_plan,
}

pub struct FftBuf<T> {
  priv data: *mut T,
  priv size: uint,
  priv capacity: uint,
}

trait TransformInput: Pod {
  fn make_buffer(capacity: uint) -> FftBuf<Self>;
  fn transform_size(input: &FftBuf<Self>) -> uint;
  fn plan(input: &FftBuf<Self>, output: &FftBuf<Cmplx<f64>>)-> fftw_plan;
}

impl TransformInput for f64 {
  #[inline]
  fn make_buffer(capacity: uint) -> FftBuf<f64> {
    unsafe {
      let _g = LOCK.lock();
      FftBuf {
        data: fftw_alloc_real(capacity as size_t),
        size: 0,
        capacity: capacity,
      }
    }
  }

  #[inline]
  fn transform_size(input: &FftBuf<f64>) -> uint {
    input.capacity/2 + 1
  }

  #[inline]
  fn plan(input: &FftBuf<f64>, output: &FftBuf<Cmplx<f64>>) -> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_r2c_1d(input.capacity as c_int, input.data, output.data, FFTW_ESTIMATE)
    }
  }
}

impl TransformInput for Cmplx<f64> {
  #[inline]
  fn make_buffer(capacity: uint) -> FftBuf<Cmplx<f64>> {
    unsafe {
      let _g = LOCK.lock();
      FftBuf {
        data: fftw_alloc_complex(capacity as size_t),
        size: 0,
        capacity: capacity,
      }
    }
  }

  #[inline]
  fn transform_size(input: &FftBuf<Cmplx<f64>>) -> uint {
    input.capacity
  }

  #[inline]
  fn plan(input: &FftBuf<Cmplx<f64>>, output: &FftBuf<Cmplx<f64>>)-> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_1d(input.capacity as c_int, input.data, output.data, FFTW_FORWARD,
                       FFTW_ESTIMATE)
    }
  }
}

trait TransformBuf {
  fn new(capacity: uint) -> Self;
  fn get_transformed_capacity(&self) -> uint;
  fn make_plan(&self, freq_dom: &FftBuf<Cmplx<f64>>) -> fftw_plan ;
}

impl<T: TransformInput> TransformBuf for FftBuf<T> {
  #[inline]
  fn new(capacity: uint) -> FftBuf<T> {
    TransformInput::make_buffer(capacity)
  }

  #[inline]
  fn get_transformed_capacity(&self) -> uint {
    TransformInput::transform_size(self)
  }

  #[inline]
  fn make_plan(&self, freq_dom: &FftBuf<Cmplx<f64>>) -> fftw_plan {
    TransformInput::plan(self, freq_dom)
  }
}

impl<T: TransformInput> FftBuf<T> {
  /// Returns a mutable view of this buffer.
  #[inline]
  pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut[T] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.data,
        len: self.size,
      })
    }
  }

  /// Returns the capacity of this buffer. The capacity is fixed, fft buffer cannot
  /// be resized.
  #[inline]
  pub fn capacity(&self) -> uint {
    self.capacity
  }

  /// Removes the last value added to the buffer.
  pub fn pop(&mut self) -> Option<T> {
    if self.size == 0 {
      None
    } else {
      self.size -= 1;
      Some(unsafe {
        *self.data.offset(self.size as int)
      })
    }
  }

  /// Adds the elements of the given slice to the fft buffer.
  /// The capacity of the transform is fixed, if there are two many values
  /// in the slice no element will be added.
  pub fn push_slice(&mut self, rhs: &[T]) -> bool {
    if self.size + rhs.len() > self.capacity {
      false
    } else {
      unsafe {
        copy_memory(self.data.offset(self.size as int), rhs.as_ptr(),
                    rhs.len());
      }
      self.size += rhs.len();
      true
    }
  }

  /// Adds a value to the buffer, if it is not already filled up.
  pub fn push(&mut self, rhs: T) -> bool {
    if self.size == self.capacity {
      false
    } else {
      unsafe {
        move_val_init(&mut *self.data.offset(self.size as int), rhs);
      }
      self.size += 1;
      true
    }
  }

  /// Creates an iterator over the values of the buffer.
  pub fn iter<'a>(&'a self) -> std::vec::Items<'a, T> {
    self.as_slice().iter()
  }
}

impl<T: TransformInput> Container for FftBuf<T> {
  /// Returns the number of values added to the buffer so far.
  fn len(&self) -> uint {
    self.size
  }
}

#[unsafe_destructor]
impl<T: TransformInput> Drop for FftBuf<T> {
  fn drop(&mut self) {
    unsafe {
      let _g = LOCK.lock();
      fftw_free(self.data as *mut c_void);
    }
  }
}

impl<T: TransformInput> Fftw<T> {
  /// Prepare a new transform for 'capacity' elements.
  /// The transform can only be computed once the input buffer is full.
  pub fn new(capacity: uint) -> Fftw<T> {
    let _time: FftBuf<T> = TransformBuf::new(capacity);
    // When the input data is real the output buffer should have a n/2 + 1 capacity,
    // n otherwise. get_transformed_capacity returns the relevant value based on the
    // type of T.
    let _freq = TransformBuf::new(_time.get_transformed_capacity());
    let _p = _time.make_plan(&_freq);
    Fftw {
      time_dom: _time,
      freq_dom: _freq,
      plan: _p,
    }
  }

  /// Prepare a new transform from the given slice of numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  #[inline]
  pub fn from_slice(slice: &[T]) -> Fftw<T> {
    let mut new = Fftw::new(slice.len());
    new.time_dom.push_slice(slice);
    new
  }

  /// Perform the actual Fourier transform computation.
  /// If the transform is not filled up yet this does nothing and return None.
  /// Otherwise, this returns an immutable view of the result. (The same you would
  /// get by calling the output() method)
  pub fn fft<'a>(&'a mut self) -> Option<&'a[Cmplx<f64>]> {
    if self.time_dom.capacity == self.time_dom.size && self.time_dom.size > 0 {
      unsafe{
        fftw_execute(self.plan);
      }
      self.freq_dom.size = self.freq_dom.capacity;
      Some(self.freq_dom.as_slice())
    } else {
      None
    }
  }

  /// Returns an immutable view of the input data.
  pub fn input<'a>(&'a self) -> &'a [T] {
    self.time_dom.as_slice()
  }

  /// Returns a mutable ref to the input buffer. You can use this to
  /// add/remove elements from the input.
  pub fn mut_input<'a>(&'a mut self) -> &'a mut FftBuf<T> {
    &mut self.time_dom
  }

  /// Returns an immutable view of the output data.
  pub fn output<'a>(&'a self) -> &'a [Cmplx<f64>] {
    self.freq_dom.as_slice()
  }

  /// Returns a mutable view of the output data.
  pub fn mut_output<'a>(&'a mut self) -> &'a mut [Cmplx<f64>] {
    self.freq_dom.as_mut_slice()
  }
}

#[unsafe_destructor]
impl<T: TransformInput> Drop for Fftw<T> {
  fn drop(&mut self) {
    unsafe {
      let _g = LOCK.lock();
      fftw_destroy_plan(self.plan);
    }
  }
}

impl<T: TransformInput> Index<uint, Option<T>> for FftBuf<T> {
  /// Returns the element in the input data, at the given index.
  fn index(&self, index: &uint) -> Option<T> {
    if *index < self.size {
      Some(unsafe {
        *self.data.offset(*index as int)
      })
    } else {
      None
    }
  }
}

impl<T: TransformInput> Vector<T> for FftBuf<T> {
  /// Returns an immutable view of the values in this buffer.
  fn as_slice<'a>(&'a self) -> &'a [T] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.data,
        len: self.size,
      })
    }
  }
}

pub mod iteration {
  use extra::complex::Cmplx;
  /// Iterator over *all* the values of a result of a transform over real data.
  pub struct HermitianItems<'a> {
    priv ptr: *Cmplx<f64>,
    priv start: *Cmplx<f64>,
    priv end: *Cmplx<f64>,
    priv dir: i8, // holds the information about the direction we are iterating in
                  // and whether there should be an even or odd number of elements.
  }

  impl<'a> Iterator<Cmplx<f64>> for HermitianItems<'a> {
    fn next(&mut self) -> Option<Cmplx<f64>> {
      if self.dir > 0 {
        if self.ptr >= self.end {
          let tmp = self.end;
          self.end = self.start;
          self.start = unsafe {
            if self.dir == 2 {
              tmp.offset(-2)
            } else {
              tmp.offset(-1)
            }
          };
          self.ptr = self.start;
          self.dir = -self.dir;
          self.next()
        } else {
          unsafe {
            let res = *self.ptr;
            self.ptr = self.ptr.offset(1);
            Some(res)
          }
        }
      } else {
        if self.ptr <= self.end {
          None
        } else {
          unsafe {
            let res = *self.ptr;
            self.ptr = self.ptr.offset(-1);
            Some(res.conj())
          }
        }
      }
    }
  }

  impl super::Fftw<f64> {
    /// Creates an iterator over *all* the values of a result from a transform over reals.
    /// Since the result of a transform over real values is a Hermitian symmetric space,
    /// only the first half of the symmetry needs to be computed. This iterator will yield
    /// the values of the result and then the remaining symmetric values.
    pub fn iter_symmetry<'a>(&'a self) -> HermitianItems<'a> {
      unsafe {
        HermitianItems {
          ptr: &*self.freq_dom.data,
          start: &*self.freq_dom.data,
          end: &*self.freq_dom.data.offset(self.freq_dom.size as int),
          dir: if self.time_dom.size % 2 == 0 {
            2
          } else {
            1
          },
        }
      }
    }
  }
}

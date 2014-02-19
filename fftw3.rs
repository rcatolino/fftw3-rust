// Copyright (c) 2014 Raphael Catolino
#[feature(macro_rules)];
#[crate_id = "fftw3_rust"];
#[crate_type = "lib"];
#[crate_type = "dylib"];

#[cfg(target_arch = "x86_64")]
extern crate extra;
extern crate num;
extern crate sync;

use num::complex::Cmplx;

use fftw3_bindgen::{FFTW_FORWARD, FFTW_BACKWARD, FFTW_ESTIMATE,
                    fftw_alloc_complex, fftw_alloc_real, fftw_free,
                    fftw_plan_dft_1d, fftw_plan_dft_r2c_1d, fftw_plan_dft_c2r_1d,
                    fftw_destroy_plan, fftw_execute, fftw_plan};

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
    use num::complex::Cmplx;
    use fftw3_rust::{Fftw, Line};

    let input = [Cmplx::new(1f64, -1f64), Cmplx::new(0f64, 2f64), Cmplx::new(4f64, 12f64)];
    let mut fftw = Fftw::from_slice(input);
    fftw.fft();
    println!("{}", Line(fftw.output()));
    ```
**/
pub struct Fftw<In, Out> {
  priv in_data: In,
  priv out_data: Out,
  priv plan: fftw_plan,
}

pub struct FftBuf<T> {
  priv data: *mut T,
  priv size: uint,
  priv capacity: uint,
}

trait TransformData: Pod {
  fn fftw_alloc(capacity: uint) -> *mut Self;
  fn transform_size(input_capacity: uint, _: &[Self]) -> uint;
  fn plan(N: uint, input: *mut Self, output: *mut Cmplx<f64>)-> fftw_plan;
  fn plan_inv(N: uint, input: *mut Cmplx<f64>, output: *mut Self)-> fftw_plan;
}

impl TransformData for f64 {
  #[inline]
  fn fftw_alloc(capacity: uint) -> *mut f64 {
    unsafe {
      let _g = LOCK.lock();
      fftw_alloc_real(capacity as size_t)
    }
  }

  #[inline]
  fn transform_size(input_capacity: uint, _: &[f64]) -> uint {
    input_capacity/2 + 1
  }

  #[inline]
  fn plan(N: uint, input: *mut f64, output: *mut Cmplx<f64>)-> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_r2c_1d(N as c_int, input, output, FFTW_ESTIMATE)
    }
  }

  #[inline]
  fn plan_inv(N: uint, input: *mut Cmplx<f64>, output: *mut f64)-> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_c2r_1d(N as c_int, input, output, FFTW_ESTIMATE)
    }
  }
}

impl TransformData for Cmplx<f64> {
  #[inline]
  fn fftw_alloc(capacity: uint) -> *mut Cmplx<f64> {
    unsafe {
      let _g = LOCK.lock();
      fftw_alloc_complex(capacity as size_t)
    }
  }

  #[inline]
  fn transform_size(input_capacity: uint, _: &[Cmplx<f64>]) -> uint {
    input_capacity
  }

  #[inline]
  fn plan(N: uint, input: *mut Cmplx<f64>, output: *mut Cmplx<f64>)-> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_1d(N as c_int, input, output, FFTW_FORWARD, FFTW_ESTIMATE)
    }
  }

  #[inline]
  fn plan_inv(N: uint, input: *mut Cmplx<f64>, output: *mut Cmplx<f64>)-> fftw_plan {
    unsafe {
      let _g = LOCK.lock();
      fftw_plan_dft_1d(N as c_int, input, output, FFTW_BACKWARD, FFTW_ESTIMATE)
    }
  }
}

trait TransformBuf<T>: Vector<T>+Container {
  fn new(capacity: uint) -> Self;
  fn get_transformed_capacity(&self) -> uint;
  fn make_plan(&mut self, out_data: &mut [Cmplx<f64>]) -> fftw_plan;
  fn make_plan_inv(&mut self, in_data: &mut [Cmplx<f64>]) -> fftw_plan;
  fn ready(&self) -> bool;
  fn as_mut_slice<'a>(&'a mut self) -> &'a mut [T];
  fn mark_filled(&mut self);
}

impl<T: TransformData> TransformBuf<T> for ~[T] {
  #[inline]
  fn new(capacity: uint) -> ~[T] {
    ::std::vec::with_capacity(capacity)
  }

  #[inline]
  fn get_transformed_capacity(&self) -> uint {
    TransformData::transform_size(self.len(), self.as_slice())
  }

  #[inline]
  fn make_plan(&mut self, out_data: &mut [Cmplx<f64>]) -> fftw_plan {
    TransformData::plan(self.len(), self.as_mut_ptr(), out_data.as_mut_ptr())
  }

  #[inline]
  fn make_plan_inv(&mut self, in_data: &mut [Cmplx<f64>]) -> fftw_plan {
    TransformData::plan_inv(self.len(), in_data.as_mut_ptr(), self.as_mut_ptr())
  }

  #[inline]
  fn ready(&self) -> bool {
    self.len() > 0
  }

  #[inline]
  fn as_mut_slice<'a>(&'a mut self) -> &'a mut[T] {
    self.as_mut_slice()
  }

  #[inline]
  fn mark_filled(&mut self) {
    let c = self.capacity();
    unsafe {
      self.set_len(c);
    }
  }
}

impl<T: TransformData> TransformBuf<T> for FftBuf<T> {
  #[inline]
  fn new(capacity: uint) -> FftBuf<T> {
    FftBuf {
      data: TransformData::fftw_alloc(capacity),
      size: 0,
      capacity: capacity,
    }
  }

  #[inline]
  fn get_transformed_capacity(&self) -> uint {
    TransformData::transform_size(self.capacity, self.as_slice())
  }

  #[inline]
  fn make_plan(&mut self, out_data: &mut [Cmplx<f64>]) -> fftw_plan {
    TransformData::plan(self.capacity, self.data, out_data.as_mut_ptr())
  }

  #[inline]
  fn make_plan_inv(&mut self, in_data: &mut [Cmplx<f64>]) -> fftw_plan {
    TransformData::plan_inv(self.capacity, in_data.as_mut_ptr(), self.data)
  }

  #[inline]
  fn ready(&self) -> bool {
    self.capacity == self.size && self.size > 0
  }

  #[inline]
  fn as_mut_slice<'a>(&'a mut self) -> &'a mut[T] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.data,
        len: self.size,
      })
    }
  }

  #[inline]
  fn mark_filled(&mut self) {
    self.size = self.capacity;
  }
}

impl<T: TransformData> FftBuf<T> {
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

impl<T: TransformData> Container for FftBuf<T> {
  /// Returns the number of values added to the buffer so far.
  fn len(&self) -> uint {
    self.size
  }
}

#[unsafe_destructor]
impl<T: TransformData> Drop for FftBuf<T> {
  fn drop(&mut self) {
    unsafe {
      let _g = LOCK.lock();
      fftw_free(self.data as *mut c_void);
    }
  }
}

impl<T: TransformData> Fftw<~[T], ~[Cmplx<f64>]> {
  /// Prepare a new transform using the given buffer. Does not perform any copy of
  /// the input data. Using a regular vector might prevent the use of simd because
  /// of alignment constraints.
  pub fn from_vec(mut vec: ~[T]) -> Fftw<~[T], ~[Cmplx<f64>]> {
    let mut _out: ~[Cmplx<f64>] = TransformBuf::new(vec.get_transformed_capacity());
    let _p = vec.make_plan(_out);
    Fftw {
      in_data: vec,
      out_data: _out,
      plan: _p,
    }
  }
}

impl<T: TransformData> Fftw<FftBuf<T>, FftBuf<Cmplx<f64>>> {
  /// Prepare a new transform from the given slice of numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  #[inline]
  pub fn from_slice(slice: &[T]) -> Fftw<FftBuf<T>, FftBuf<Cmplx<f64>>> {
    let mut new: Fftw<FftBuf<T>, FftBuf<Cmplx<f64>>> = Fftw::new(slice.len());
    new.in_data.push_slice(slice);
    new
  }

  /// Returns a mutable ref to the input buffer. You can use this to
  /// add/remove elements from the input.
  #[inline]
  pub fn ref_input<'a>(&'a mut self) -> &'a mut FftBuf<T> {
    &mut self.in_data
  }

  /// Prepare a new transform for 'capacity' elements.
  /// The transform can only be computed once the input buffer is full.
  pub fn new(capacity: uint) -> Fftw<FftBuf<T>, FftBuf<Cmplx<f64>>> {
    let mut _in: FftBuf<T> = TransformBuf::new(capacity);
    // When the input data is real the output buffer should have a n/2 + 1 capacity,
    // n otherwise. get_transformed_capacity returns the relevant value based on the
    // type of T.
    let mut _out: FftBuf<Cmplx<f64>> = TransformBuf::new(_in.get_transformed_capacity());
    let _p = _in.make_plan(_out.as_mut_slice());
    Fftw {
      in_data: _in,
      out_data: _out,
      plan: _p,
    }
  }
}

impl Fftw<FftBuf<Cmplx<f64>>, FftBuf<Cmplx<f64>>> {
  /// Prepare a new transform from the given slice of numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  #[inline]
  pub fn from_slice_inv(slice: &[Cmplx<f64>]) -> Fftw<FftBuf<Cmplx<f64>>, FftBuf<Cmplx<f64>>> {
    let mut new = Fftw::new_inv(slice.len());
    new.in_data.push_slice(slice);
    new
  }

  /// Prepare a new inverse transform for 'capacity' elements.
  /// The transform can only be computed once the input buffer is full.
  /// For complex->real transform, the complex data should only have n/2+1
  /// elements (the first part of the Hermitian symmetry).
  pub fn new_inv(capacity: uint) -> Fftw<FftBuf<Cmplx<f64>>, FftBuf<Cmplx<f64>>> {
    let mut _out: FftBuf<Cmplx<f64>> = TransformBuf::new(capacity);
    let mut _in: FftBuf<Cmplx<f64>> = TransformBuf::new(_out.get_transformed_capacity());
    let _p = _out.make_plan_inv(_in.as_mut_slice());
    Fftw {
      in_data: _in,
      out_data: _out,
      plan: _p,
    }
  }
}

impl Fftw<FftBuf<Cmplx<f64>>, FftBuf<f64>> {
  /// Prepare a new transform from the given slice of numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  #[inline]
  pub fn from_slice_c2r(slice: &[Cmplx<f64>]) -> Fftw<FftBuf<Cmplx<f64>>, FftBuf<f64>> {
    let mut new = Fftw::new_c2r(slice.len());
    new.in_data.push_slice(slice);
    new
  }


  /// Prepare a new inverse transform for 'capacity' elements.
  /// The transform can only be computed once the input buffer is full.
  /// For complex->real transform, the complex data should only have n/2+1
  /// elements (the first part of the Hermitian symmetry).
  pub fn new_c2r(capacity: uint) -> Fftw<FftBuf<Cmplx<f64>>, FftBuf<f64>> {
    let mut _out: FftBuf<f64> = TransformBuf::new(capacity);
    let mut _in: FftBuf<Cmplx<f64>> = TransformBuf::new(_out.get_transformed_capacity());
    let _p = _out.make_plan_inv(_in.as_mut_slice());
    Fftw {
      in_data: _in,
      out_data: _out,
      plan: _p,
    }
  }
}

impl<Tin: TransformData, In: TransformBuf<Tin>,
     Tout: TransformData, Out: TransformBuf<Tout>> Fftw<In, Out> {
  /// Perform the actual Fourier transform computation.
  /// If the transform is not filled up yet this does nothing and return None.
  /// Otherwise, this returns an immutable view of the result. (The same you would
  /// get by calling the output() method)
  pub fn compute<'a>(&'a mut self) -> Option<&'a[Tout]> {
    if self.in_data.ready() {
      unsafe{
        fftw_execute(self.plan);
      }
      self.out_data.mark_filled();
      Some(self.out_data.as_slice())
    } else {
      None
    }
  }

  #[inline]
  /// Returns an immutable view of the input data.
  pub fn input<'a>(&'a self) -> &'a [Tin] {
    self.in_data.as_slice()
  }

  #[inline]
  /// Returns an immutable view of the input data.
  pub fn mut_input<'a>(&'a mut self) -> &'a mut [Tin] {
    self.in_data.as_mut_slice()
  }

  #[inline]
  /// Returns an immutable view of the output data.
  pub fn output<'a>(&'a self) -> &'a [Tout] {
    self.out_data.as_slice()
  }

  #[inline]
  /// Returns a mutable view of the output data.
  pub fn mut_output<'a>(&'a mut self) -> &'a mut [Tout] {
    self.out_data.as_mut_slice()
  }
}

#[unsafe_destructor]
impl<In, Out> Drop for Fftw<In, Out> {
  fn drop(&mut self) {
    unsafe {
      let _g = LOCK.lock();
      fftw_destroy_plan(self.plan);
    }
  }
}

impl<T: TransformData> Index<uint, Option<T>> for FftBuf<T> {
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

impl<T: TransformData> Vector<T> for FftBuf<T> {
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
  use num::complex::Cmplx;
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

  impl super::Fftw<super::FftBuf<f64>, super::FftBuf<Cmplx<f64>>> {
    /// Creates an iterator over *all* the values of a result from a transform over reals.
    /// Since the result of a transform over real values is a Hermitian symmetric space,
    /// only the first half of the symmetry needs to be computed. This iterator will yield
    /// the values of the result and then the remaining symmetric values.
    pub fn iter_symmetry<'a>(&'a self) -> HermitianItems<'a> {
      unsafe {
        HermitianItems {
          ptr: &*self.out_data.data,
          start: &*self.out_data.data,
          end: &*self.out_data.data.offset(self.out_data.size as int),
          dir: if self.in_data.size % 2 == 0 {
            2
          } else {
            1
          },
        }
      }
    }
  }
}

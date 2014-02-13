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
    let mut fftw = Fftw::from_slice_real(input);
    fftw.fft();
    println!("{}", Line(fftw.result()));
    ```

    Compute the 1d transform for complex values :

    ```rust
    use extra::complex::Cmplx;
    use fftw3_rust::{Fftw, Line};

    let input = [Cmplx::new(1f64, -1f64), Cmplx::new(0f64, 2f64), Cmplx::new(4f64, 12f64)];
    let mut fftw = Fftw::from_slice(input);
    fftw.fft();
    println!("{}", Line(fftw.result()));
    ```
**/
pub struct Fftw<T> {
  priv inp: *mut T,
  priv outp: *mut Cmplx<f64>,
  priv size: uint,
  priv out_size: uint,
  priv plan: fftw_plan,
  priv capacity: uint,
  priv out_capacity: uint,
}

impl Fftw<f64> {
  /// Prepare a new real transform for 'capacity' elements.
  /// The transform can only be computed once all the elements have been added.
  pub fn new_real(capacity: uint) -> Fftw<f64> {
    unsafe {
      let _g = LOCK.lock();
      let inp = fftw_alloc_real(capacity as size_t);
      let outp = fftw_alloc_complex((capacity/2 + 1) as size_t);
      Fftw {
        inp : inp,
        outp: outp,
        plan: fftw_plan_dft_r2c_1d(capacity as c_int, inp, outp, FFTW_ESTIMATE),
        size: 0,
        out_size: 0,
        capacity: capacity,
        out_capacity: capacity/2+1,
      }
    }
  }

  /// Prepare a new transform from the given slice of real numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  pub fn from_slice_real(slice: &[f64]) -> Fftw<f64> {
    let mut new = Fftw::new_real(slice.len());
    new.push_slice(slice);
    new
  }
}

impl Fftw<Cmplx<f64>> {
  /// Prepare a new complex transform for 'capacity' elements.
  /// The transform can only be computed once all the elements have been added.
  pub fn new(capacity: uint) -> Fftw<Cmplx<f64>> {
    unsafe {
      let _g = LOCK.lock();
      let inp = fftw_alloc_complex(capacity as size_t);
      let outp = fftw_alloc_complex(capacity as size_t);
      Fftw {
        inp : inp,
        outp: outp,
        plan: fftw_plan_dft_1d(capacity as c_int, inp, outp,
                               FFTW_FORWARD, FFTW_ESTIMATE),
        size: 0,
        out_size: 0,
        capacity: capacity,
        out_capacity: capacity,
      }
    }
  }

  /// Prepare a new transform from the given slice of complex numbers.
  /// The elements of the slice are copied in an internal buffer allocated by fftw3.
  pub fn from_slice(slice: &[Cmplx<f64>]) -> Fftw<Cmplx<f64>> {
    let mut new = Fftw::new(slice.len());
    new.push_slice(slice);
    new
  }
}

impl<T: Pod> Fftw<T> {
  /// Returns an immutable view of the result of a previous computation.
  /// If no transform has been computed yet, returns an empty slice.
  pub fn result<'a>(&'a self) -> &'a [Cmplx<f64>] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.outp,
        len: self.out_size,
      })
    }
  }

  /// Returns an mutable view of the result of a previous computation.
  /// If no transform has been computed yet, returns an empty slice.
  pub fn mut_result<'a>(&'a mut self) -> &'a mut [Cmplx<f64>] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.outp,
        len: self.out_size,
      })
    }
  }

  /// Returns a mutable view of the input values added to the transform so far.
  pub fn as_mut_slice<'a>(&'a mut self) -> &'a mut[T] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.inp,
        len: self.size,
      })
    }
  }

  /// Returns the number of values that the transform expects.
  pub fn capacity(&self) -> uint {
    self.capacity
  }

  /// Removes the last value added to the transform.
  pub fn pop(&mut self) -> Option<T> {
    if self.size == 0 {
      None
    } else {
      self.size -= 1;
      Some(unsafe {
        *self.inp.offset(self.size as int)
      })
    }
  }

  /// Adds the elements of the given slice to the input of the transform.
  /// The capacity of the transform is fixed, if there are two many values
  /// in the slice no element will be added.
  pub fn push_slice(&mut self, rhs: &[T]) -> bool {
    if self.size + rhs.len() > self.capacity {
      false
    } else {
      unsafe {
        copy_memory(self.inp.offset(self.size as int), rhs.as_ptr(),
                    rhs.len());
      }
      self.size += rhs.len();
      true
    }
  }

  /// Adds a value to the transform, if it is not already filled up.
  pub fn push(&mut self, rhs: T) -> bool {
    if self.size == self.capacity {
      false
    } else {
      unsafe {
        move_val_init(&mut *self.inp.offset(self.size as int), rhs);
      }
      self.size += 1;
      true
    }
  }

  /// Perform the actual Fourier transform computation.
  /// If the transform is not filled up yet this does nothing and return None.
  /// Otherwise, this returns an immutable view of the result. (The same you would
  /// get by calling the result() method)
  pub fn fft<'a>(&'a mut self) -> Option<&'a[Cmplx<f64>]> {
    if self.capacity == self.size && self.size > 0 {
      unsafe{
        fftw_execute(self.plan);
      }
      self.out_size = self.out_capacity;
      Some(self.result())
    } else {
      None
    }
  }
}

impl<T: Pod> Container for Fftw<T> {
  /// Returns the number of values added to the transform so far.
  fn len(&self) -> uint {
    self.size
  }
}

#[unsafe_destructor]
impl<T: Pod> Drop for Fftw<T> {
  fn drop(&mut self) {
    unsafe {
      let _g = LOCK.lock();
      fftw_destroy_plan(self.plan);
      fftw_free(self.inp as *mut c_void);
      fftw_free(self.outp as *mut c_void);
    }
  }
}

impl<T: Pod> Index<uint, Option<T>> for Fftw<T> {
  /// Returns the element in the input data, at the given index.
  fn index(&self, index: &uint) -> Option<T> {
    if *index < self.size {
      Some(unsafe {
        *self.inp.offset(*index as int)
      })
    } else {
      None
    }
  }
}

impl<T> Vector<T> for Fftw<T> {
  /// Returns an immutable view of the values added to the transform so far.
  fn as_slice<'a>(&'a self) -> &'a [T] {
    unsafe{
      transmute(CplxSlice {
        data: &*self.inp,
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
    /// Creates an iterator over *all* the values of the result of the last real transform.
    /// If no transform has been computed yet the iterator won't yield any value.
    pub fn iter<'a>(&'a self) -> HermitianItems<'a> {
      unsafe {
        HermitianItems {
          ptr: &*self.outp,
          start: &*self.outp,
          end: &*self.outp.offset(self.out_size as int),
          dir: if self.size % 2 == 0 {
            2
          } else {
            1
          },
        }
      }
    }
  }
}

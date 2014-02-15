// Copyright (c) 2014 Raphael Catolino
#[macro_escape];

#[macro_export]
macro_rules! hra {
  ($($item:expr),+) => (
    ~[$($item as f64),+]
  )
}

#[macro_export]
macro_rules! ra {
  ($($item:expr),+) => (
    [$($item as f64),+]
  )
}

#[macro_export]
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

#[macro_export]
macro_rules! c {
  ($re:expr , $im:expr) => {
    Cmplx::new($re as f64, $im as f64)
  };
  ($re:expr) => {
    Cmplx::new($re as f64, 0f64)
  }
}



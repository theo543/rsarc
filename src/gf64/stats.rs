#![allow(dead_code)]

use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

struct Count(AtomicUsize);

impl Count {
    const fn new() -> Self {
        Count(AtomicUsize::new(0))
    }

    fn load(&self) -> usize {
        self.0.load(Ordering::Relaxed)
    }

    fn add(&self, value: usize) {
        self.0.fetch_add(value, Ordering::Relaxed);
    }
}

pub struct ThreadLocalCount {
    count: Cell<usize>,
    destination: &'static Count,
}

pub trait LocalKeyExt {
    fn increment(&'static self);
}

impl LocalKeyExt for std::thread::LocalKey<ThreadLocalCount> {
    fn increment(&'static self) {
        self.with(|local| local.count.set(local.count.get() + 1));
    }
}

impl Drop for ThreadLocalCount {
    fn drop(&mut self) {
        self.destination.add(self.count.get());
    }
}

impl From<&'static Count> for ThreadLocalCount {
    fn from(destination: &'static Count) -> Self {
        ThreadLocalCount { count: Cell::new(0), destination }
    }
}

pub struct DummyCount;
impl DummyCount {
    pub fn increment(&self) {}
}

#[cfg(feature = "gf64_stats")]
macro_rules! def_counts {
    ( $( $x:ident; )* ) => {
        mod statics {
            $(pub static $x: super::Count = super::Count::new();)*
        }
        pub mod thread_locals {
            pub use super::LocalKeyExt;
            thread_local! {
                $(pub static $x: super::ThreadLocalCount = super::ThreadLocalCount::from(&super::statics::$x);)*
            }
        }
        #[allow(non_snake_case)]
        mod stat_getters {
            $(pub fn $x() -> usize {
                super::statics::$x.load() + super::thread_locals::$x.with(|local| local.count.get())
            })*
        }
    };
}

#[cfg(not(feature = "gf64_stats"))]
macro_rules! def_counts {
    ( $( $x:ident; )* ) => {
        pub mod thread_locals {
            $(pub static $x: super::DummyCount = super::DummyCount;)*
        }
    };
}

def_counts! {
    MULTIPLICATIONS_PERFORMED;
    MULTIPLICATIONS_IN_INVERSION;
    INVERSES_COMPUTED;
    DIVISION_ITERATIONS;
    EUCLIDEAN_ITERATIONS;
}

fn add_separators(mut value: usize) -> String {
    let mut result = format!("{:03}", value % 1000);
    value /= 1000;
    while value >= 1000 {
        result = format!("{:03}_{}", value % 1000, result);
        value /= 1000;
    }
    format!("{value}_{result}")
}

fn print_sep(name: &str, value: usize) {
    println!("{}{}", name, add_separators(value));
}

#[cfg(feature = "gf64_stats")]
pub fn print_stats() {
    use stat_getters::*;
    print_sep("Inverses computed: ", INVERSES_COMPUTED());
    print_sep("Multiplications performed as part of inversion: ", MULTIPLICATIONS_IN_INVERSION());
    print_sep("Multiplications performed not as part of inversion: ", MULTIPLICATIONS_PERFORMED() - MULTIPLICATIONS_IN_INVERSION());
    print_sep("Total multiplications performed: ", MULTIPLICATIONS_PERFORMED());
    print_sep("Division iterations: ", DIVISION_ITERATIONS());
    print_sep("Euclidean iterations: ", EUCLIDEAN_ITERATIONS());
}

#[cfg(not(feature = "gf64_stats"))]
pub fn print_stats() {}

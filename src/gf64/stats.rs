use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct Count(AtomicUsize);

impl Count {
    const fn new() -> Self {
        Count(AtomicUsize::new(0))
    }

    pub fn load(&self) -> usize {
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


macro_rules! def_counts {
    ( $( $x:ident; )* ) => {
        mod statics {
            $(pub static $x: super::Count = super::Count::new();)*
        }
        pub mod thread_locals {
            thread_local! {
                $(pub static $x: super::ThreadLocalCount = super::ThreadLocalCount::from(&super::statics::$x);)*
            }
        }
        #[allow(non_snake_case)]
        pub mod stat_getters {
            $(pub fn $x() -> usize {
                super::statics::$x.load() + super::thread_locals::$x.with(|local| local.count.get())
            })*
        }
    };
}

def_counts! {
    MULTIPLICATIONS_PERFORMED;
    MULTIPLICATIONS_IN_DIVISION;
    INVERSES_COMPUTED;
    DIVISION_ITERATIONS;
    EUCLIDEAN_ITERATIONS;
}

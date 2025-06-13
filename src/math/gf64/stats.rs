#![cfg_attr(not(feature = "gf64_stats"), allow(dead_code))] // code will be dead if stats are disabled

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
    fn load(&'static self) -> usize;
}

impl LocalKeyExt for std::thread::LocalKey<ThreadLocalCount> {
    fn increment(&'static self) {
        self.with(|local| local.count.set(local.count.get() + 1));
    }
    fn load(&'static self) -> usize {
        self.with(|local| local.count.get())
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

#[cfg(feature = "gf64_stats")]
// Using a macro for this is a bit overkill, but I wanted to learn macro_rules :)
macro_rules! def_counts {
    ( $($label:literal: $x:tt;)* ) => {

        def_counts!(@defs {} $( $x )*);

        pub fn print_stats() {
            use stat_getters::*;
            $(print_sep($label, def_counts!(@brackets $x));)*
        }

    };

    (@brackets $x:ident) => { $x() }; // add brackets to call getter corresponding to stat $x
    (@brackets { $x:expr }) => { $x }; // leave expr as is

    // filter out exprs before defining statics, thread locals, getters
    (@defs { $($state:tt)* } $next:ident $($tail:tt)*) => { def_counts!(@defs { $($state)* $next } $($tail)*); };
    (@defs $state:tt $_:tt $($tail:tt)*) => { def_counts!(@defs $state $($tail)*); };

    // done filtering, now definitions
    (@defs { $( $x:ident )* } ) => {
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
    ( $( $_:literal: $x:tt; )* ) => {

        pub struct DummyCount;
        impl DummyCount {
            pub fn increment(&self) {}
        }

        pub mod thread_locals {
            $( def_counts!($x); )*
        }

        pub fn print_stats() {}

    };
    ($x:ident) => { pub static $x: super::DummyCount = super::DummyCount; };
    ($_:expr) => {};
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
    println!("{}: {}", name, add_separators(value));
}

// Define the stats using the macro selecting by #[cfg].
// If the feature "gf64_stats" is disabled, dummy statics are defined, and print_stats() is a no-op.

def_counts! {
    "Inverses computed": INVERSES_COMPUTED;
    "Multiplications performed as part of inversion": MULTIPLICATIONS_IN_INVERSION;
    "Multiplications performed not as part of inversion": { MULTIPLICATIONS_PERFORMED() - MULTIPLICATIONS_IN_INVERSION() };
    "Total multiplications performed": MULTIPLICATIONS_PERFORMED;
    "Division iterations": DIVISION_ITERATIONS;
    "Euclidean iterations": EUCLIDEAN_ITERATIONS;
}

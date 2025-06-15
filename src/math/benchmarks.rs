use std::hint::black_box;

use super::gf64::{GF64, invert_by_pow};

fn timed(name: &str, size: usize, f: impl FnOnce()) {
    let start = std::time::Instant::now();
    f();
    let duration = start.elapsed();
    println!("{name},{},{}", duration.as_nanos(), duration.as_nanos() as f64 / size as f64);
}

fn randomize(buf: &mut [GF64]) {
    for x in buf {
        x.0 = fastrand::u64(..);
    }
}

pub fn math_benchmarks(size: usize) {
    let start = std::time::Instant::now();
    let mut buf1 = vec![GF64(0); size];
    let mut buf2 = vec![GF64(0); size];
    randomize(&mut buf1);
    randomize(&mut buf2);
    timed("multiplication", size, || {
        for i in 0..size {
            buf1[i] *= buf2[i];
        }
    });
    black_box(&buf1);
    randomize(&mut buf1);
    timed("inversion", size, || {
        for x in &mut buf1 {
            *x = x.invert();
        }
    });
    black_box(&buf1);
    randomize(&mut buf1);
    timed("inversion_by_pow", size, || {
        for x in &mut buf1 {
            *x = invert_by_pow(*x);
        }
    });
    black_box(&buf1);
    eprintln!("Total time to run benchmarks: {:?}", start.elapsed());
}

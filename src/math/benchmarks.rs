use std::hint::black_box;

use super::gf64::{GF64, invert_by_pow};

const SIZE: usize = 1 << 22; // 4 MiB

fn timed(name: &str, f: impl FnOnce()) {
    let start = std::time::Instant::now();
    f();
    let duration = start.elapsed();
    println!("{name},{},{}", duration.as_nanos(), duration.as_nanos() as f64 / SIZE as f64);
}

fn randomize(buf: &mut [GF64]) {
    for x in buf {
        x.0 = fastrand::u64(..);
    }
}

pub fn math_benchmarks() {
    let mut buf1 = vec![GF64(0); SIZE];
    let mut buf2 = vec![GF64(0); SIZE];
    randomize(&mut buf1);
    randomize(&mut buf2);
    timed("multiplication", || {
        for i in 0..SIZE {
            buf1[i] *= buf2[i];
        }
    });
    black_box(&buf1);
    randomize(&mut buf1);
    timed("inversion", || {
        for x in &mut buf1 {
            *x = x.invert();
        }
    });
    black_box(&buf1);
    randomize(&mut buf1);
    timed("inversion_by_pow", || {
        for x in &mut buf1 {
            *x = invert_by_pow(*x);
        }
    });
    black_box(&buf1);
}

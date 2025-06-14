use std::time::{Duration, Instant};

use crate::math::{gf64::GF64, novelpoly::{compute_error_locator_poly, formal_derivative, forward_transform, inverse_transform, precompute_derivative_factors, precompute_transform_factors, DerivativeFactors, TransformFactors}};

mod math;

#[inline(never)]
fn oversample(buf: &mut [GF64], t_factors: &TransformFactors, t_factors_offset: &TransformFactors) {
    let size = buf.len();
    assert!(size.is_power_of_two());
    let (data, parity) = buf.split_at_mut(buf.len() / 2);
    parity.copy_from_slice(data);
    inverse_transform(parity, t_factors);
    forward_transform(parity, t_factors_offset);
}

#[inline(never)]
fn recover(buf: &mut [GF64], errors: &[u64], t_factors: &TransformFactors, d_factors: &DerivativeFactors, err: &[GF64], err_derivative_inverse: &[GF64]) {
    let size = buf.len();
    assert!(size.is_power_of_two());
    assert_eq!(errors.len(), size / 2);
    for (x, y) in buf.iter_mut().zip(err.iter()) {
        *x *= *y;
    }
    inverse_transform(buf, t_factors);
    formal_derivative(buf, d_factors);
    forward_transform(buf, t_factors);
    for e in errors {
        buf[*e as usize] *= err_derivative_inverse[*e as usize];
    }
}

#[inline(never)]
fn timed(fn_: impl FnOnce(), size: usize, name: &str) {
    let start = Instant::now();
    fn_();
    let len = Instant::now() - start;
    println!("{name}, {size}, {}", len.as_nanos());
}

fn randomize(buf: &mut [GF64]) {
    for x in buf.iter_mut() {
        x.0 = fastrand::u64(..);
    }
}

#[inline(never)]
fn precompute(size: usize, errors: &[u64]) -> (TransformFactors, TransformFactors, DerivativeFactors, Vec<GF64>, Vec<GF64>) {
    let t_factors = precompute_transform_factors(size.ilog2(), GF64(0));
    let t_factors_offset = precompute_transform_factors(size.ilog2(), GF64(size as u64 / 2));
    let t_factors_large = precompute_transform_factors(size.ilog2(), GF64(0));
    let d_factors = precompute_derivative_factors(size.ilog2());
    let (err, mut err_derivative) = compute_error_locator_poly(errors, size, &t_factors_large, &d_factors);
    for x in err_derivative.iter_mut() {
        *x = x.invert();
    }
    (t_factors, t_factors_offset, d_factors, err, err_derivative)
}

fn main() {
    math::gf64::detect_cpu_features();

    for size in (8..=17).map(|x| (1 << x) as usize) {
        let mut errors = (0..size as u64).collect::<Vec<_>>();
        fastrand::shuffle(&mut errors);
        errors.truncate(size / 2);

        let (t_factors, t_factors_offset, d_factors, err, err_derivative) = precompute(size, &errors);
        let mut buf = vec![GF64(0); size];
        let mut buf_2 = vec![GF64(0); size];
        for _ in 0..100 {
            randomize(&mut buf);
            timed(|| oversample(&mut buf, &t_factors, &t_factors_offset), size, "oversample");
            buf_2.copy_from_slice(&buf);
            for e in &errors {
                buf_2[*e as usize].0 = fastrand::u64(..);
            }
            timed(|| recover(&mut buf_2, &errors, &t_factors, &d_factors, &err, &err_derivative), size, "recover");
            for e in &errors {
                assert_eq!(buf[*e as usize], buf_2[*e as usize]);
            }
        }
    }
}

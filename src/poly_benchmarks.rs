use std::{hint::black_box, time::Instant};
use super::math;

use crate::math::{gf64::GF64, novelpoly::{compute_error_locator_poly, formal_derivative, forward_transform, inverse_transform, precompute_derivative_factors, precompute_transform_factors, DerivativeFactors, TransformFactors}};

fn multiply_by_x_plus_a(poly: &mut [GF64], a: GF64) {
    assert!(poly.len() > 1);
    // The highest degree must be 0 to avoid overflow.
    assert!(*poly.last().unwrap() == GF64(0));
    poly[poly.len() - 1] = poly[poly.len() - 2];
    for i in (1..poly.len() - 1).rev() {
        poly[i] = poly[i] * a + poly[i - 1];
    }
    poly[0] *= a;
}

fn add_poly_with_mul(dest: &mut [GF64], src: &[GF64], mul: GF64) {
    assert_eq!(dest.len(), src.len());
    for (d, s) in dest.iter_mut().zip(src) {
        *d += *s * mul;
    }
}

fn update_divided_difference(div_diff: &mut [GF64], y: GF64, i: usize) {
    div_diff[i] = y;
    for j in (0..i).rev() {
        div_diff[j] = (div_diff[j+1] + div_diff[j]) / (GF64(i as u64) + GF64(j as u64));
    }
}

pub fn newton_interpolation(y_values: &[GF64], poly: &mut [GF64], memory: &mut [GF64]) {
    assert_eq!(y_values.len(), poly.len());
    assert!(memory.len() >= y_values.len() * 2);
    let (basis_poly, remaining_memory) = memory.split_at_mut(y_values.len());
    let div_diff = &mut remaining_memory[..y_values.len()];
    basis_poly[0] = GF64(1);
    basis_poly[1..].fill(GF64(0));
    div_diff[0] = y_values[0];
    div_diff[1..].fill(GF64(0));
    poly[0] = y_values[0];
    poly[1..].fill(GF64(0));
    for i in 1..y_values.len() {
        multiply_by_x_plus_a(&mut basis_poly[..=i], GF64((i - 1) as u64));
        update_divided_difference(div_diff, y_values[i], i);
        add_poly_with_mul(&mut poly[..=i], &basis_poly[..=i], div_diff[0]);
    }
}

pub fn evaluate_poly(poly: &[GF64], x: GF64) -> GF64 {
    let mut result = *poly.last().unwrap();
    for coefficient in poly.iter().rev().skip(1) {
        result *= x;
        result += *coefficient;
    }
    result
}

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
fn oversample_newton(data: &mut [GF64], parity: &mut [GF64], memory: &mut [GF64]) {
    let (poly, memory) = memory.split_at_mut(data.len());
    newton_interpolation(data, poly, memory);
    for (i, p) in parity.iter_mut().enumerate() {
        *p = evaluate_poly(poly, GF64(i as u64));
    }
}

#[inline(never)]
fn timed<T>(fn_: impl FnOnce() -> T, size: usize, name: &str) -> T {
    let start = Instant::now();
    let x = fn_();
    let len = Instant::now() - start;
    println!("{name}, {size}, {}", len.as_nanos());
    x
}

fn randomize(buf: &mut [GF64]) {
    for x in buf.iter_mut() {
        x.0 = fastrand::u64(..);
    }
}

fn precompute_oversampling(size: usize) -> (TransformFactors, TransformFactors) {
    (precompute_transform_factors(size.ilog2(), GF64(0)), precompute_transform_factors(size.ilog2(), GF64(size as u64)))
}

fn precompute_recovery(size: usize, errors: &[u64]) -> (TransformFactors, DerivativeFactors, Vec<GF64>, Vec<GF64>) {
    let t_factors = precompute_transform_factors(size.ilog2(), GF64(0));
    let d_factors = precompute_derivative_factors(size.ilog2());
    let (err, mut err_derivative) = compute_error_locator_poly(errors, size, &t_factors, &d_factors);
    for x in err_derivative.iter_mut() {
        *x = x.invert();
    }
    (t_factors, d_factors, err, err_derivative)
}

pub fn poly_benchmarks() {
    math::gf64::detect_cpu_features();

    for size in (8..=14).map(|x| (1 << x) as usize) {
        let mut errors = (0..size as u64).collect::<Vec<_>>();
        fastrand::shuffle(&mut errors);
        errors.truncate(size / 2);
        let mut buf = vec![GF64(0); size];
        let mut buf_2 = vec![GF64(0); size];
        let mut memory = vec![GF64(0); size * 3];
        for _ in 0..5 {
            randomize(&mut buf);
            timed(|| oversample_newton(&mut buf, &mut buf_2, &mut memory), size, "oversample_with_newton");
            black_box(&buf_2);
        }
    }

    for size in (8..=17).map(|x| (1 << x) as usize) {
        let mut errors = (0..size as u64).collect::<Vec<_>>();
        fastrand::shuffle(&mut errors);
        errors.truncate(size / 2);

        for _ in 0..=49 {
            black_box(timed(|| precompute_oversampling(size / 2), size, "precompute_oversampling"));
            black_box(timed(|| precompute_recovery(size, &errors), size, "precompute_recovery"));
        }

        let (t_factors, t_factors_offset) = timed(|| precompute_oversampling(size / 2), size, "precompute_oversampling");
        let (t_factors_large, d_factors, err, err_derivative) = timed(|| precompute_recovery(size, &errors), size, "precompute_recovery");

        let mut buf = vec![GF64(0); size];
        let mut buf_2 = vec![GF64(0); size];
        for _ in 0..500 {
            randomize(&mut buf);
            timed(|| oversample(&mut buf, &t_factors, &t_factors_offset), size, "oversample");
            buf_2.copy_from_slice(&buf);
            for e in &errors {
                buf_2[*e as usize].0 = fastrand::u64(..);
            }
            timed(|| recover(&mut buf_2, &errors, &t_factors_large, &d_factors, &err, &err_derivative), size, "recovery");
            for e in &errors {
                assert_eq!(buf[*e as usize], buf_2[*e as usize]);
            }
        }
    }
}

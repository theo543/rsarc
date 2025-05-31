//! Implementation of algorithms from the paper "Novel Polynomial Basis and Its Application to Reed-Solomon Erasure Codes"

// In the following comments, ω_i refers to the integer i interpreted as an element of GF(2^64), which in code would be `GF64(i)`.

use super::gf64::{GF64, u64_as_gf64};

fn eval_vanishing_poly(x: GF64, j: u32) -> GF64 {
    //! The subspace vanishing polynomial W_j is zero for points in the subspace {ω_0, ω_1, ... ω_{2^j - 1}}.
    //! W_j(x) = x * (x + ω_1) * ... (x + ω_{2^j - 1})
    //! Computing this is O(2^j). The maximum j used for encoding or decoding data of length n is log_2(n), so the complexity is at most O(n).
    assert!(j < 64);
    let mut result = GF64(1);
    for i in 0..(1 << j) {
        result *= x + GF64(i);
    }
    result
}

fn get_normalization_factor(j: u32) -> GF64 {
    //! Computes normalization factor 1 / W_j(ω_{2^j}) with internal caching.

    use std::sync::{atomic::{AtomicU64, Ordering}, Mutex};
    static CACHE: [AtomicU64; 64] = [const { AtomicU64::new(0) }; 64];
    static COMPUTE_IN_PROGRESS: [Mutex<()>; 64] = [const { Mutex::new(()) }; 64];

    let mut factor = CACHE[j as usize].load(Ordering::Relaxed);
    if factor == 0 {
        // lock prevents two threads computing same factor at the same time (which would waste CPU time)
        let _lock = COMPUTE_IN_PROGRESS[j as usize].lock().unwrap();
        factor = CACHE[j as usize].load(Ordering::SeqCst);
        if factor == 0 {
            factor = eval_vanishing_poly(GF64(1 << j), j).invert().0;
            CACHE[j as usize].store(factor, Ordering::SeqCst);
        }
    }
    GF64(factor)
}

fn eval_normalized_vanishing_poly(x: GF64, j: u32) -> GF64 {
    //! Normalized subspace vanishing polynomial evaluates to 1 at the point ω_{2^j}, which is helpful for the transform.
    eval_vanishing_poly(x, j) * get_normalization_factor(j)
}

pub struct TransformFactors {
    pow: u32,
    offset: GF64,
    factors: Box<[GF64]>,
}

pub fn precompute_transform_factors(len: u64, offset: GF64) -> TransformFactors {
    //! Precomputes twiddle factors used for the forward and inverse transforms.
    //! There are 1 + 2 + ... + n / 2 = n - 1 unique factors for a transform of size n.
    assert!(len.is_power_of_two());
    let pow = len.ilog2();
    let mut factors = vec![GF64(0); (len - 1).try_into().unwrap()].into_boxed_slice();
    let mut factor_idx = 0;
    for step in 0..pow {
        let groups = 1 << (pow - step - 1);
        for group in 0..groups {
            factors[factor_idx] = eval_normalized_vanishing_poly(GF64(group * (1 << (step + 1))) + offset, step);
            factor_idx += 1;
        }
    }
    TransformFactors { pow, offset, factors }
}

pub fn inverse_transform(data: &mut [GF64], &TransformFactors {pow, ref factors, ..}: &TransformFactors) {
    //! Converts data from evaluations of a polynomial at points ω_{0 + offset}, ω_{1 + offset}, ..., ω_{n - 1 + offset} to coefficients in the non-standard basis.
    assert_eq!(data.len(), 1 << pow);
    let mut factor_idx = 0;
    for step in 0..pow {
        let group_len = 1 << step;
        let groups = 1 << (pow - step - 1);
        for group in 0..groups {
            for x in 0..group_len {
                let a = (group * group_len * 2 + x) as usize;
                let b = a + group_len as usize;
                data[b] += data[a]; // b = a' + b' = a' + a' + b = b
                data[a] += data[b] * factors[factor_idx]; // a = a' + factor * b = a + factor * b + factor * b = a
            }
            factor_idx += 1;
        }
    }
}

pub fn forward_transform(data: &mut [GF64], &TransformFactors {pow, ref factors, ..}: &TransformFactors) {
    //! Converts data back from coefficients in the non-standard basis to evaluations.
    assert_eq!(data.len(), 1 << pow);
    let mut factor_idx = factors.len();
    for step in (0..pow).rev() {
        let group_len = 1 << step;
        let groups = 1 << (pow - step - 1);
        for group in (0..groups).rev() {
            for x in 0..group_len {
                let a = (group * group_len * 2 + x) as usize;
                let b = a + group_len as usize;
                data[a] += factors[factor_idx - 1] * data[b]; // a' = a + factor * b
                data[b] += data[a]; // b' = a + (factor + 1) * b = a + factor * b + b = a' + b
            }
            factor_idx -= 1;
        }
    }
}

pub struct DerivativeFactors(Box<[GF64]>);

pub fn precompute_derivative_factors(len: usize) -> DerivativeFactors {
    //! Precomputes factors needed for computing the formal derivative in the non-standard basis.
    assert!(len < 64);
    let mut factors = vec![GF64(1); len].into_boxed_slice();
    for l in 1..len {
        // factors[l] = ω_{1} * ω_{2} * ... * ω_{2^l - 1} / W_l(ω_{2^l})
        for j in (1 << (l - 1))..(1 << l) {
            factors[l] *= GF64(j);
        }
        if l + 1 != len {
            factors[l + 1] = factors[l];
        }
        factors[l] *= get_normalization_factor(l as u32);
    }
    DerivativeFactors(factors)
}

pub fn formal_derivative(data: &mut [GF64], DerivativeFactors(factors): &DerivativeFactors) {
    //! Computes the formal derivative of a polynomial with coefficients in the non-standard basis.
    assert!(data.len().is_power_of_two());
    let max_bit = data.len().ilog2() as usize;
    assert!(max_bit < factors.len() + 1);
    for i in 0..data.len() {
        // Iterate over bits in i
        for (set_bit_idx, set_bit) in (0..=max_bit).map(|bit_idx| (bit_idx, 1 << bit_idx)).filter(|(_, bit)| i & bit != 0) {
            data[i - set_bit] += data[i] * factors[set_bit_idx];
        }
        data[i] = GF64(0);
    }
}

pub fn compute_error_locator_poly(locations: &[u64], out_len: usize, t_factors: &[TransformFactors], d_factors: &DerivativeFactors) -> (Vec<GF64>, Vec<GF64>) {
    //! Computes values of the error locator polynomial (x + e_0) * (x + e_1) * ... * (x + e_n) and values of the formal derivative in O(n log^2 n) time.
    assert!(!locations.is_empty());
    assert!(out_len >= locations.len());
    assert!(out_len.is_power_of_two());
    assert_eq!(t_factors.len(), out_len.ilog2() as usize);

    let locations = u64_as_gf64(locations);

    for (i, f) in t_factors.iter().rev().enumerate() {
        assert_eq!(f.pow, i as u32 + 1);
        assert_eq!(f.offset, GF64(0));
    }

    fn rec(locations: &[GF64], out_len: usize, factors: &[TransformFactors], depth: usize, out_values: Option<&mut Vec<GF64>>) -> Vec<GF64> {
        assert_ne!(locations.len(), 0);
        assert!(out_len > locations.len());

        if locations.len() == 1 {
            if let Some(out_values) = out_values {
                *out_values = (0..out_len).map(|i| locations[0] + GF64(i as u64)).collect();
            }

            return [locations[0], GF64(1)].iter().copied().chain(std::iter::repeat_n(GF64(0), out_len - 2)).collect();
        }

        // When compute_error_locator_poly is called with locations.len() > out_len / 2, sometimes locations.len() + 1 will equal out_len.
        // In this case, splitting into two would cause locations.len() == out_len in one of the branches, so the extra value must be multiplied in later.
        let special_case = locations.len() + 1 == out_len;

        let mut a = rec(&locations[..locations.len() / 2], out_len / 2, factors, depth + 1, None);
        a.resize(out_len, GF64(0));

        let mut b = rec(&locations[locations.len() / 2..locations.len() - special_case as usize], out_len / 2, factors, depth + 1, None);
        b.resize(out_len, GF64(0));

        forward_transform(&mut a, &factors[depth]);
        forward_transform(&mut b, &factors[depth]);

        for (x, y) in a.iter_mut().zip(b.iter().copied()) {
            *x *= y;
        }

        if special_case {
            let extra_value = *locations.last().unwrap();
            for (i, x) in a.iter_mut().enumerate() {
                *x *= GF64(i as u64) + extra_value;
            }
        }

        if let Some(out_values) = out_values {
            b.copy_from_slice(&a);
            *out_values = b;
        }

        inverse_transform(&mut a, &factors[depth]);
        a
    }

    let mut values = vec![];
    let mut coefficients = rec(locations, out_len, t_factors, 0, Some(&mut values));
    for c in &coefficients[locations.len() + 1..] {
        assert_eq!(*c, GF64(0));
    }
    assert_eq!(values.len(), coefficients.len());

    // Take derivative of coefficients and evaluate
    formal_derivative(&mut coefficients, d_factors);
    forward_transform(&mut coefficients, &t_factors[0]);

    (values, coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math::gf64::tests::{gf64, gf64_array}, utils::IntoU64Ext};

    #[test]
    fn roundtrip() {
        fn test<const N: usize>() {
            let offset = gf64();
            let factors = precompute_transform_factors(N.as_u64(), offset);
            let data = gf64_array::<N>();
            let mut roundtripped = data;
            let mut roundtripped2 = data;

            for _ in 0..10 {
                forward_transform(&mut roundtripped, &factors);
                inverse_transform(&mut roundtripped2, &factors);
            }

            for _ in 0..10 {
                inverse_transform(&mut roundtripped, &factors);
                forward_transform(&mut roundtripped2, &factors);            
            }

            assert_eq!(data, roundtripped);
            assert_eq!(data, roundtripped2);
        }

        test::<1>();
        test::<2>();
        test::<4>();
        test::<8>();
        test::<64>();
        test::<128>();
    }

    fn eval<const N: usize, const M: usize>(poly: [GF64; N], offset: GF64) -> [GF64; M] {
        std::array::from_fn(|i| {
            let mut result = poly[N - 1];
            for coefficient in poly.iter().copied().rev().skip(1) {
                result *= GF64(i as u64) + offset;
                result += coefficient;
            }
            result
        })
    }

    #[test]
    fn agrees_with_standard_eval() {
        let poly = gf64_array::<128>();
        let offset = gf64();
        let offset2 = gf64();
        let mut values: [GF64; 128] = eval(poly, offset);
        inverse_transform(&mut values, &precompute_transform_factors(128, offset));
        forward_transform(&mut values, &precompute_transform_factors(128, offset2));
        assert_eq!(values, eval(poly, offset2));
    }

    #[test]
    fn excess_coeff_are_zero() {
        let poly = gf64_array::<128>();
        let offset = gf64();
        let mut values: [GF64; 512] = eval(poly, offset);
        inverse_transform(&mut values, &precompute_transform_factors(512, offset));
        assert_eq!(values[128..], [GF64(0); 512 - 128]);
    }

    #[test]
    fn oversampled_has_same_coeff() {
        let offset = gf64();
        let mut poly = gf64_array::<128>();
        let mut bigger = [GF64(0); 256];
        bigger[..128].copy_from_slice(&poly);
        bigger[128..].copy_from_slice(&poly);
        let base_factors = precompute_transform_factors(128, offset);
        inverse_transform(&mut poly, &base_factors);
        inverse_transform(&mut bigger[128..], &base_factors);
        forward_transform(&mut bigger[128..], &precompute_transform_factors(128, offset + GF64(128)));
        inverse_transform(&mut bigger, &precompute_transform_factors(256, offset));
        assert_eq!(bigger[..128], poly);
        assert_eq!(bigger[128..], [GF64(0); 128]);
    }

    fn standard_formal_derivative<const N: usize>(poly: [GF64; N]) -> [GF64; N] {
        // In GF(2^n), c * x^j has derivative c * x^{j - 1} if j is odd, else 0.
        let mut derivative = [GF64(0); N];
        for i in (1..N).step_by(2) {
            derivative[i - 1] = poly[i];
        }
        derivative
    }

    #[test]
    fn agrees_with_standard_formal_derivative() {
        let poly = gf64_array::<128>();
        let offset = gf64();
        let offset2 = gf64();
        let mut values: [GF64; 128] = eval(poly, offset);
        inverse_transform(&mut values, &precompute_transform_factors(128, offset));
        formal_derivative(&mut values, &precompute_derivative_factors(8));
        forward_transform(&mut values, &precompute_transform_factors(128, offset2));
        assert_eq!(values, eval(standard_formal_derivative(poly), offset2));
    }

    const N: usize = 512;
    const M: usize = 64;
    const F: usize = N.ilog2() as usize;
    fn test_data_recovery(error_count: usize) {

        let mut errors: [u64; N] = std::array::from_fn(|i| i as u64);
        fastrand::shuffle(&mut errors);

        let factors: [TransformFactors; F] = std::array::from_fn(|i| precompute_transform_factors(1 << (F - i), GF64(0)));
        let d_factors = precompute_derivative_factors(F);
        let (error_values, error_derivative_values) = compute_error_locator_poly(&errors[..error_count], N, &factors, &d_factors);

        let errors = || errors.iter().take(error_count).map(|e| *e as usize);

        for e in errors() {
            assert_eq!(error_values[e], GF64(0));
        }

        let data: [GF64; M] = gf64_array();
        let mut encoded = [GF64(0); N];
        encoded[..M].copy_from_slice(&data);
        for i in (M..N).step_by(M) {
            let slice = &mut encoded[i..i + M];
            slice.copy_from_slice(&data);
            inverse_transform(slice, &factors[F - M.ilog2() as usize]);
            forward_transform(slice, &precompute_transform_factors(M.as_u64(), GF64(i.as_u64())));
        }
        let backup = encoded;

        // Simulate data loss
        for e in errors() {
            encoded[e] = GF64(0);
        }

        // Multiply by error locator
        for (x, y) in encoded.iter_mut().zip(error_values.iter().copied()) {
            *x *= y;
        }

        // Take the derivative
        inverse_transform(&mut encoded, &factors[0]);
        formal_derivative(&mut encoded, &d_factors);
        forward_transform(&mut encoded, &factors[0]);

        for e in errors() {
            // By the product rule, (f(x) * e(x))' = f'(x) * e(x) + f(x) * e'(x)
            // At error points, e(x) = 0, so (f(x) * e(x))' = f(x) * e'(x) <=> f(x) = (f(x) * e(x))' / e'(x)

            // e'(e_idx) cannot be zero because e(x) has a simple root at e_idx, i.e. it can written as e(x) = (x + e_idx) * g(x) where g(e_idx) != 0
            // This implies e'(e_idx) != 0 because:
            //  e'(x) = (x + e_idx)' * g(x) + (x + e_idx) * g'(x) = g(x) + (x + e_idx) * g'(x)
            //  e'(e_idx) = g(e_idx) + (e_idx + e_idx) * g'(x) = g(e_idx) != 0
            // This is true in all fields including GF(2^64).

            assert_eq!(encoded[e] / error_derivative_values[e], backup[e]);
        }
    }

    #[test]
    fn can_recover_data() {
        test_data_recovery(1);
        test_data_recovery(64);
        test_data_recovery(199);
        test_data_recovery(N - M); // maximum number of errors that can be corrected
    }

    #[test]
    #[should_panic]
    fn cannot_recover_from_too_many_errors() {
        test_data_recovery(N - M + 1); // cannot recover, as the degree of f * e is N + 1, so interpolating with N values will almost certainly give the wrong result
    }
}

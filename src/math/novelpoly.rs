//! Implementation of algorithms from the paper "Novel Polynomial Basis and Its Application to Reed-Solomon Erasure Codes"

use super::gf64::GF64;

fn eval_vanishing_poly(x: GF64, j: u32) -> GF64 {
    //! The subspace vanishing polynomial W_j is zero for points in the subspace {F_0, F_1, ... F_{2^j - 1}}.
    //! W_i(x) = x * (x + F_1) * ... (x + F_{2^j - 1})
    //! Computing this is O(2^j). The maximum j used for encoding or decoding data of length n is log_2(n), so the complexity is at most O(n).
    assert!(j < 64);
    let mut result = GF64(1);
    for i in 0..(1 << j) {
        result *= x + GF64(i);
    }
    result
}

fn eval_normalized_vanishing_poly(x: GF64, i: u32) -> GF64 {
    //! Normalized subspace vanishing polynomial is one at the point F_{2^j}, which is helpful for the transform.
    eval_vanishing_poly(x, i) / eval_vanishing_poly(GF64(1 << i), i)
}

pub struct TransformFactors {
    pow: u32,
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
    TransformFactors { pow, factors }
}

pub fn inverse_transform(data: &mut [GF64], &TransformFactors {pow, ref factors}: &TransformFactors) {
    //! Converts data from evaluations of a polynomial at points F_{0 + offset}, F_{1 + offset}, ..., F_{n - 1 + offset} to coefficients in the non-standard basis.
    assert_eq!(data.len(), 1 << pow);
    let mut factor_idx = 0;
    for step in 0..pow {
        let group_len = 1 << (step + 1);
        let groups = 1 << (pow - step - 1);
        for group in 0..groups {
            for x in 0..group_len / 2 {
                let i = (group * group_len + x) as usize;
                let j = i + (group_len / 2) as usize;
                data[j] += data[i]; // j = i' + j' = i' + i' + j = j
                data[i] += data[j] * factors[factor_idx]; // i = i' + factor * j = i + factor * j + factor * j = i
            }
            factor_idx += 1;
        }
    }
}

pub fn forward_transform(data: &mut [GF64], &TransformFactors {pow, ref factors}: &TransformFactors) {
    //! Converts data back from coefficients in the non-standard basis to evaluations.
    assert_eq!(data.len(), 1 << pow);
    let mut factor_idx = factors.len();
    for step in (0..pow).rev() {
        let group_len = 1 << (step + 1);
        let groups = 1 << (pow - step - 1);
        for group in (0..groups).rev() {
            for x in 0..group_len / 2 {
                let i = (group * group_len + x) as usize;
                let j = i + (group_len / 2) as usize;
                data[i] += factors[factor_idx - 1] * data[j]; // i' = i + factor * j
                data[j] += data[i]; // j' = i + (factor + 1) * j = i + factor * j + j = i' + j
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
        // factors[l] = F_{1} * F_{2} * ... * F_{2^l - 1} / W_i(F_{2^i})
        for j in (1 << (l - 1))..(1 << l) {
            factors[l] *= GF64(j);
        }
        if l + 1 != len {
            factors[l + 1] = factors[l];
        }
        factors[l] /= eval_vanishing_poly(GF64(1 << l), l as u32);
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

#[cfg(test)]
mod tests {
    use crate::{math::{gf64::{tests::{gf64, gf64_array}, GF64}, novelpoly::{forward_transform, inverse_transform, precompute_transform_factors}, polynomials::evaluate_poly}, utils::IntoU64Ext};

    use super::{formal_derivative, precompute_derivative_factors};

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
        std::array::from_fn(|i| evaluate_poly(&poly, GF64(i.as_u64()) + offset))
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
        let values2: [GF64; 128] = eval(standard_formal_derivative(poly), offset2);
        assert_eq!(values, values2);
    }
}

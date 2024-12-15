use crate::gf64::GF64;

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

fn update_divided_difference(div_diff: &mut [GF64], x_values: &[GF64], y: GF64, i: usize) {
    div_diff[i] = y;
    for j in (0..i).rev() {
        div_diff[j] = (div_diff[j+1] + div_diff[j]) / (x_values[i] + x_values[j]);
    }
}

fn update_divided_difference_fixed_x_values(div_diff: &mut [GF64], y: GF64, i: usize) {
    div_diff[i] = y;
    for j in (0..i).rev() {
        div_diff[j] = (div_diff[j+1] + div_diff[j]) / (GF64(i as u64) + GF64(j as u64));
    }
}

pub fn newton_interpolation(y_values: &[GF64], x_values: Option<&[GF64]>, poly: &mut [GF64], memory: &mut [GF64]) {
    if let Some(x_values) = x_values { assert_eq!(y_values.len(), x_values.len()); }
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
        if let Some(x_values) = x_values {
            multiply_by_x_plus_a(&mut basis_poly[..=i], x_values[i - 1]);
            update_divided_difference(div_diff, x_values, y_values[i], i);
        } else {
            multiply_by_x_plus_a(&mut basis_poly[..=i], GF64((i - 1) as u64));
            update_divided_difference_fixed_x_values(div_diff, y_values[i], i);
        }
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

#[cfg(test)]
mod tests{
    use crate::gf64::{tests::{gf64, gf64_array}, GF64};
    use super::{evaluate_poly, newton_interpolation};

    fn distinct_gf64_array<const N: usize>() -> [GF64; N] {
        loop {
            let array = gf64_array::<N>();
            let mut sorted = array.map(|gf| gf.0);
            sorted.sort();
            let distinct = sorted.windows(2).all(|w| w[0] != w[1]);
            if distinct { return array };
        }
    }

    fn shuffle_together(a: &mut [GF64], b: &mut [GF64]) {
        assert_eq!(a.len(), b.len());
        let mut rng = fastrand::Rng::new();
        let mut dup_rng = rng.clone();
        rng.shuffle(a);
        dup_rng.shuffle(b);
    }

    #[test]
    fn constant_poly() {
        let x = gf64();
        assert_eq!(evaluate_poly(&[x], gf64()), x);
        assert_eq!(evaluate_poly(&[x.0, 0, 0, 0, 0, 0, 0].map(GF64), gf64()), x);
    }

    #[test]
    fn identity_poly() {
        let x = gf64();
        assert_eq!(evaluate_poly(&[GF64(0), GF64(1)], x), x);
        assert_eq!(evaluate_poly(&[0, 1, 0, 0, 0, 0, 0].map(GF64), x), x);
    }

    #[test]
    fn can_roundtrip_interpolate_eval() {
        // Fill memory with random data to check the function doesn't expect it to be zeroed.
        let mut memory = gf64_array::<100>();
        let y_vals: [GF64; 50] = gf64_array();
        let x_vals: [GF64; 50] = distinct_gf64_array();
        let mut poly = gf64_array::<50>();
        newton_interpolation(&y_vals, Some(&x_vals), &mut poly, &mut memory);
        let roundtrip = x_vals.map(|x| evaluate_poly(&poly, x));
        assert_eq!(y_vals, roundtrip);
    }

    #[test]
    fn can_roundtrip_fixed_x_values() {
        let mut memory = gf64_array::<100>();
        let y_vals: [GF64; 50] = gf64_array();
        let mut poly = gf64_array::<50>();
        newton_interpolation(&y_vals, None, &mut poly, &mut memory);
        let x_values: [GF64; 50] = std::array::from_fn(|x| GF64(x as u64));
        let roundtrip = x_values.map(|x| evaluate_poly(&poly, x));
        assert_eq!(y_vals, roundtrip);
    }

    #[test]
    fn can_recover_lost_data() {
        let mut memory = gf64_array::<100>();
        let original_data: [GF64; 50] = gf64_array();

        let mut all_x_values: [GF64; 100] = distinct_gf64_array();
        let original_x_values: [GF64; 50] = (&all_x_values[..50]).try_into().unwrap();

        let mut original_poly = [GF64(0); 50];
        newton_interpolation(&original_data, Some(&original_x_values), &mut original_poly, &mut memory);

        let mut all_y_values = [GF64(0); 100];
        let (data, parity) = all_y_values.split_at_mut(50);
        data.copy_from_slice(&original_data);
        parity.iter_mut().zip(&all_x_values[50..]).for_each(|(y, x)| *y = evaluate_poly(&original_poly, *x));

        shuffle_together(&mut all_x_values, &mut all_y_values);

        let mut recovered_poly = [GF64(0); 50];
        newton_interpolation(&all_y_values[..50], Some(&all_x_values[..50]), &mut recovered_poly, &mut memory);
        let recovered_data = original_x_values.map(|x| evaluate_poly(&recovered_poly, x));

        assert_eq!(original_poly, recovered_poly);
        assert_eq!(original_data, recovered_data);
    }

    #[test]
    #[should_panic]
    fn insufficient_memory() {
        newton_interpolation(&[GF64(0); 10], None, &mut [GF64(0); 10], &mut [GF64(0); 19]);
    }
}

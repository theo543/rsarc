mod gf64;
use gf64::GF64;

use fastrand::Rng;

fn main() {
    // Quick test of GF64
    let mut rng = Rng::new();
    let mut values: [GF64; 1000] = std::array::from_fn(|_| GF64(rng.u64(0..)));
    let mut product = GF64(1);
    for value in values {
        product *= value;
    }
    // Galois field multiplication should be commutative.
    rng.shuffle(&mut values);
    for value in values {
        product /= value;
    }
    assert_eq!(product.0, 1);
}

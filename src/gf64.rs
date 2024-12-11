use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

#[derive(Clone, Copy)]
pub(crate) struct GF64(pub u64);

// Prime polynomial for GF(2^64), x^64 term not included.
// x^64 + x^4 + x^3 + x + 1
const POLY: u64 = 16 + 8 + 2 + 1;

impl Add for GF64 {
    type Output = GF64;

    fn add(self, rhs: Self) -> GF64 {
        GF64(self.0 ^ rhs.0)
    }
}

impl Mul for GF64 {
    type Output = GF64;

    fn mul(self, GF64(mut rhs): Self) -> GF64 {
        let GF64(mut lhs) = self;
        let mut product: u64 = 0;
        for _ in 0..=63 {
            if rhs & 1 == 1 { product ^= lhs; }
            rhs >>= 1;
            let carry = lhs >> 63;
            lhs <<= 1;
            if carry == 1 { lhs ^= POLY; }
        }
        GF64(product)
    }
}

impl AddAssign for GF64 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl MulAssign for GF64 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl GF64 {
    pub fn invert(self) -> GF64 {
        // Invert by raising to power 2^64 - 2 = 2^63 + 2^62 + ... + 2
        let mut result = self * self;
        let mut pow = result;
        for _ in 2..=63 {
            pow *= pow;
            result *= pow;
        }
        result
    }
}

impl Div for GF64 {
    type Output = GF64;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.invert()
    }
}

impl DivAssign for GF64 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

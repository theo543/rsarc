use std::cell::Cell;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

pub static MULTIPLICATIONS_PERFORMED: AtomicUsize = AtomicUsize::new(0);
pub static MULTIPLICATIONS_IN_INVERSION: AtomicUsize = AtomicUsize::new(0);
pub static INVERSES_COMPUTED: AtomicUsize = AtomicUsize::new(0);

struct ThreadLocalCount {
    count: Cell<usize>,
    destination: &'static AtomicUsize,
}

impl ThreadLocalCount {
    fn add(&self, value: usize) {
        self.count.set(self.count.get() + value);
    }
    fn increment(&self) {
        self.add(1);
    }
}

impl Drop for ThreadLocalCount {
    fn drop(&mut self) {
        self.destination.fetch_add(self.count.get(), Ordering::Relaxed);
    }
}

impl From<&'static AtomicUsize> for ThreadLocalCount {
    fn from(destination: &'static AtomicUsize) -> Self {
        ThreadLocalCount { count: Cell::new(0), destination }
    }
}

// Temporary for testing, remove after optimizing division to reduce multiplications.
thread_local! {
    static MULTIPLICATIONS_PERFORMED_LOCAL: ThreadLocalCount = ThreadLocalCount::from(&MULTIPLICATIONS_PERFORMED);
    static MULTIPLICATIONS_IN_DIVISION_LOCAL: ThreadLocalCount = ThreadLocalCount::from(&MULTIPLICATIONS_IN_INVERSION);
    static INVERSES_COMPUTED_LOCAL: ThreadLocalCount = ThreadLocalCount::from(&INVERSES_COMPUTED);
}

#[derive(PartialEq, Eq, Clone, Copy)]
#[repr(transparent)] // <- to allow safe conversion
pub struct GF64(pub u64);

impl Debug for GF64 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#064b}", self.0)
    }
}

pub fn u64_as_gf64(slice: &[u64]) -> &[GF64] {
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const GF64, slice.len()) }
}

pub fn u64_as_gf64_mut(slice: &mut [u64]) -> &mut [GF64] {
    unsafe { std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut GF64, slice.len()) }
}

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
        MULTIPLICATIONS_PERFORMED_LOCAL.with(|local| local.increment());
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
        assert_ne!(self, GF64(0));
        INVERSES_COMPUTED_LOCAL.with(|local| local.increment());
        // one multiplication for initializing result, 2 multiplications per 63 - 2 + 1 = 62 loop iterations
        MULTIPLICATIONS_IN_DIVISION_LOCAL.with(|local| local.add(1 + 2 * 62));
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

#[cfg(test)]
pub mod tests {
    use super::GF64;

    pub fn gf64() -> GF64 {
        GF64(fastrand::u64(0..))
    }

    pub fn gf64_array<const N: usize>() -> [GF64; N] {
        std::array::from_fn(|_| gf64())
    }

    fn product(values: &[GF64]) -> GF64 {
        let mut product = GF64(1);
        for value in values {
            product *= *value;
        }
        product
    }

    #[test]
    fn addition_commutative() {
        let a = gf64();
        let b = gf64();
        assert_eq!(a + b, b + a);
    }

    #[test]
    fn multiplication_commutative() {
        let a = gf64();
        let b = gf64();
        assert_eq!(a * b, b * a);
    }

    #[test]
    fn invert() {
        let a = gf64();
        let b = gf64();
        assert_eq!(a * a.invert(), GF64(1));
        assert_eq!(a * b * b.invert(), a);
        assert_eq!(a / b, a * b.invert());
    }

    #[test]
    fn invert_many() {
        let mut values = gf64_array::<1000>();
        let prod = product(&values);
        fastrand::shuffle(&mut values);
        let inverses = values.map(GF64::invert);
        let inv_prod = product(&inverses[1..]);
        assert_eq!(prod * inv_prod, values[0]);
    }

    #[test]
    #[should_panic]
    fn cannot_invert_zero() {
        GF64(0).invert();
    }
}

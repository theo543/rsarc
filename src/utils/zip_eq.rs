//! Like the crate `zip_eq`, but doesn't use the unstable `TrustedLen` trait.

use std::iter::Zip;

pub trait ZipEqExt<A, B> {
    fn zip_eq(self, other: A) -> Zip<Self, B> where Self: Sized;
}

impl<A: IntoIterator<IntoIter = B>, B: ExactSizeIterator, C: ExactSizeIterator> ZipEqExt<A, B> for C {
    #[inline]
    fn zip_eq(self, other: A) -> Zip<Self, B> {
        //! `zip` which asserts that the iterators have the same length.
        let other = other.into_iter();
        assert_eq!(self.len(), other.len());
        self.zip(other)
    }
}

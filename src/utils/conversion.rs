pub trait IntoU64Ext {
    fn as_u64(&self) -> u64;
}

impl IntoU64Ext for usize {
    fn as_u64(&self) -> u64 { u64::try_from(*self).unwrap() }
}

pub trait IntoUSizeExt {
    fn as_usize(&self) -> usize;
}

impl IntoUSizeExt for u64 {
    fn as_usize(&self) -> usize { usize::try_from(*self).expect("u64 value too large for usize - maybe you are trying to process a large file on a 32-bit system?") }
}

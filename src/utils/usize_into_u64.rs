pub trait IntoU64Ext {
    fn as_u64(&self) -> u64;
}

impl IntoU64Ext for usize {
    fn as_u64(&self) -> u64 { u64::try_from(*self).unwrap() }
}

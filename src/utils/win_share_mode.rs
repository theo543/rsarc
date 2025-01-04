use std::fs::OpenOptions;

pub trait ShareModeExt {
    fn share_mode_lock(&mut self) -> &mut Self;
}

#[cfg(windows)]
impl ShareModeExt for OpenOptions {
    fn share_mode_lock(&mut self) -> &mut Self {
        // don't allow other processes to delete or write to the file
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_SHARE_READ: u32 = 0x00000001;
        self.share_mode(FILE_SHARE_READ);
        self
    }
}

#[cfg(not(windows))]
impl ShareModeExt for OpenOptions {
    // not supported
    fn share_mode_lock(&mut self) -> &mut Self { self }
}

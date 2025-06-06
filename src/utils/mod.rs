pub mod progress;

mod win_share_mode;
pub use win_share_mode::ShareModeExt;

mod conversion;
pub use conversion::IntoU64Ext;
pub use conversion::IntoUSizeExt;

mod zip_eq;
pub use zip_eq::ZipEqExt;

use sysinfo::{System, RefreshKind, MemoryRefreshKind};

pub fn get_available_memory() -> u64 {
    let sys_mem = System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::nothing().with_ram()));
    sys_mem.available_memory()
}

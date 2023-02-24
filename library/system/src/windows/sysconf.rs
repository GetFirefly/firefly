use core::mem;

use crate::sync::OnceLock;

static SYSTEM_INFO: OnceLock<SystemInfo> = OnceLock::new();

#[inline]
pub fn page_size() -> usize {
    SYSTEM_INFO.get_or_init(SystemInfo::get).page_size
}

#[inline]
pub fn num_cpus() -> usize {
    SYSTEM_INFO.get_or_init(SystemInfo::get).num_cpus
}

/// A friendly representation of the Windows system information struct
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SystemInfo {
    page_size: usize,
    allocation_granularity: usize,
    num_cpus: usize,
}
impl SystemInfo {
    pub(crate) unsafe fn get() -> Self {
        use winapi::um::sysinfoapi::GetSystemInfo;

        let mut info = mem::zeroed();
        GetSystemInfo(&mut info);
        SystemInfo {
            page_size: info.dwPageSize as usize,
            num_cpus: info.dwNumberOfProcessors as usize,
            allocation_granularity: info.dwAllocationGranularity as usize,
        }
    }
}

/// A friendly representation of the Windows memory information struct
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MemoryInfo {
    used: usize,
    total_physical: usize,
    available_physical: usize,
    total_virtual: usize,
    available_virtual: usize,
    available_extended_virtual: usize,
}
impl MemoryInfo {
    #[allow(unused)]
    pub(crate) unsafe fn get() -> Self {
        use winapi::um::sysinfoapi::GlobalMemoryStatusEx;

        let mut info = mem::zeroed();
        GlobalMemoryStatusEx(&mut info);
        MemoryInfo {
            used: info.dwMemoryLoad as usize,
            total_physical: info.ullTotalPhys as usize,
            available_physical: info.ullAvailPhys as usize,
            total_virtual: info.ullTotalVirtual as usize,
            available_virtual: info.ullAvailVirtual as usize,
            available_extended_virtual: info.ullAvailExtendedVirtual as usize,
        }
    }
}

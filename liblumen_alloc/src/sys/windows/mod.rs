mod malloc;
pub use malloc::{
    alloc,
    alloc_zeroed,
    realloc,
    free,
};

mod mmap;

#[inline]
pub(crate) fn pagesize() -> usize {
    use winapi::um::sysinfoapi::GetSystemInfo;

    let mut info = core::mem::zeroed();
    GetSystemInfo(&mut info);
    info.dwPageSize
}

pub fn get_num_cpus() -> usize {
    unsafe {
        let mut info = core::mem::zeroed();
        GetSystemInfo(&mut info);
        info.dwNumberOfProcessors as usize

    }
}

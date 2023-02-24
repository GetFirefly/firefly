pub mod alloc;
pub mod mmap;
mod sysconf;

pub use self::sysconf::*;
pub use std::time;

pub mod threading {
    #![allow(unused)]

    use winapi::um::processthreadsapi::GetCurrentThreadId;
    use winapi::um::processthreadsapi::{GetCurrentProcess, GetProcessId};
    use winapi::um::processthreadsapi::{TlsAlloc, TlsFree, TlsGetValue, TlsSetValue};
    use winapi::um::winnt::HANDLE;

    /// This struct represents key information about the current process
    pub struct ProcessInfo {
        p: HANDLE,
        pid: usize,
    }
    impl ProcessInfo {
        /// Loads information about the current process, returning a `ProcessInfo` struct
        pub(crate) unsafe fn get() -> Self {
            let p = GetCurrentProcess();
            let pid = GetProcessId(p) as usize;
            Self { p, pid }
        }

        /// Returns the current thread ID of the caller
        #[inline]
        pub fn current_thread_id() -> usize {
            unsafe { GetCurrentThreadId() as usize }
        }

        /// Gets the current process ID
        #[inline(always)]
        pub fn process_id(&self) -> usize {
            self.pid
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub struct ThreadLocalKey(u32);
    impl ThreadLocalKey {
        #[inline]
        pub unsafe fn new() -> Self {
            Self(TlsAlloc())
        }

        #[inline]
        pub unsafe fn delete(self) -> bool {
            TlsFree(self.0) == 0
        }

        #[inline]
        pub unsafe fn get<T>(&self) -> *mut T {
            TlsGetValue(self.0) as *mut _ as *mut T
        }

        #[inline]
        pub unsafe fn set<T>(&self, value: *const T) {
            TlsSetValue(self.0, value as *mut _);
        }
    }
}

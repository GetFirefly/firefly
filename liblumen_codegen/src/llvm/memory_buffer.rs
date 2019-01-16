use std::ffi::CStr;
use std::fs::File;
use std::path::Path;

use llvm_sys::core::LLVMGetBufferStart;
use llvm_sys::prelude::LLVMMemoryBufferRef;

pub enum MemoryBuffer {
    Ref(LLVMMemoryBufferRef),
    Pointer(*mut libc::c_char),
}
impl MemoryBuffer {
    pub fn from_ref(buf: LLVMMemoryBufferRef) -> MemoryBuffer {
        MemoryBuffer::Ref(buf)
    }

    pub fn from_ptr(buf: *mut libc::c_char) -> MemoryBuffer {
        MemoryBuffer::Pointer(buf)
    }

    pub fn to_string(&self) -> String {
        let start = match *self {
            MemoryBuffer::Ref(buf) => unsafe { LLVMGetBufferStart(buf) as *mut libc::c_char },
            MemoryBuffer::Pointer(buf) => buf,
        };
        let s = unsafe { CStr::from_ptr(start) };
        s.to_string_lossy().into_owned()
    }

    pub fn write_to_file(&self, path: &Path) -> Result<(), std::io::Error> {
        use std::io::Write;

        let mut f = File::create(path)?;
        let start = match *self {
            MemoryBuffer::Ref(buf) => unsafe { LLVMGetBufferStart(buf) as *mut libc::c_char },
            MemoryBuffer::Pointer(buf) => buf,
        };
        let s = unsafe { CStr::from_ptr(start) };
        f.write_all(s.to_bytes())?;
        Ok(())
    }
}

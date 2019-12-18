use std::io;
use std::slice;

use llvm_sys::core::{LLVMGetBufferStart, LLVMGetBufferSize, LLVMDisposeMemoryBuffer};

pub type MemoryBufferRef = llvm_sys::prelude::LLVMMemoryBufferRef;

#[derive(Debug)]
pub struct MemoryBuffer(MemoryBufferRef);

impl MemoryBuffer {
    #[allow(unused)]
    pub fn new(buffer: MemoryBufferRef) -> Self {
        assert!(!buffer.is_null());
        Self(buffer)
    }


    /// Gets a byte slice of this `MemoryBuffer`.
    #[allow(unused)]
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let start = LLVMGetBufferStart(self.0);
            slice::from_raw_parts(start as *const _, self.get_size())
        }
    }

    /// Gets the byte size of this `MemoryBuffer`.
    pub fn get_size(&self) -> usize {
        unsafe {
            LLVMGetBufferSize(self.0)
        }
    }

    /// Copies the buffer to the given writer.
    #[allow(unused)]
    pub fn copy_to<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.as_slice())
    }
}

impl Drop for MemoryBuffer {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeMemoryBuffer(self.0)
        }
    }
}

use std::convert::{AsMut, AsRef};
use std::ffi::CString;
use std::io;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::path::Path;
use std::ptr;
use std::slice;

use anyhow::anyhow;

use llvm_sys::core::{LLVMDisposeMemoryBuffer, LLVMGetBufferSize, LLVMGetBufferStart};

use super::strings::LLVMString;

pub type MemoryBufferRef = llvm_sys::prelude::LLVMMemoryBufferRef;

#[derive(Debug)]
pub struct MemoryBuffer<'a> {
    buffer: MemoryBufferRef,
    _phantom: PhantomData<&'a llvm_sys::LLVMMemoryBuffer>,
}

impl<'a> MemoryBuffer<'a> {
    /// Wrap an existing memory buffer reference
    pub fn new(buffer: MemoryBufferRef) -> Self {
        assert!(!buffer.is_null());
        Self {
            buffer,
            _phantom: PhantomData,
        }
    }

    /// Consumes this wrapper and returns the raw `LLVMMemoryBufferRef`
    ///
    /// This is used when handing ownership of the buffer to a function in LLVM,
    /// requiring us to intentionally "leak" the `MemoryBuffer` wrapper so that
    /// the drop implementation is not run.
    pub fn into_mut(self) -> MemoryBufferRef {
        let buffer = self.buffer;
        mem::forget(self);
        buffer
    }

    /// Creates a MemoryBuffer with the contents of the file at the given path
    pub fn create_from_file<P: AsRef<Path>>(file: P) -> anyhow::Result<MemoryBuffer<'static>> {
        use llvm_sys::core::LLVMCreateMemoryBufferWithContentsOfFile;

        let file = file.as_ref();
        if !file.exists() || !file.is_file() {
            return Err(anyhow!(
                "unable to create memory buffer from {}: not a file",
                file.to_string_lossy()
            ));
        }
        let file = file.to_str().ok_or_else(|| {
            anyhow!("unable to create memory buffer from file: path is not valid utf-8")
        })?;

        let path = CString::new(file)
            .map_err(|e| anyhow!("unable to create memory buffer from file: {}", e))?;

        let mut buffer = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();

        let failed = unsafe {
            LLVMCreateMemoryBufferWithContentsOfFile(
                path.as_ptr() as *const libc::c_char,
                &mut buffer,
                err_string.as_mut_ptr(),
            )
        };

        if failed != 0 {
            let err_string = unsafe { err_string.assume_init() };
            return Err(anyhow!(
                "unable to create memory buffer from file: {}",
                LLVMString::new(err_string)
            ));
        }

        Ok(MemoryBuffer::new(buffer))
    }

    /// Creates a MemoryBuffer with the contents read from stdin
    #[allow(unused)]
    pub fn create_from_stdin() -> anyhow::Result<MemoryBuffer<'static>> {
        use llvm_sys::core::LLVMCreateMemoryBufferWithSTDIN;

        let mut buffer = ptr::null_mut();
        let mut err_string = MaybeUninit::uninit();

        let failed =
            unsafe { LLVMCreateMemoryBufferWithSTDIN(&mut buffer, err_string.as_mut_ptr()) };

        if failed != 0 {
            let err_string = unsafe { err_string.assume_init() };
            return Err(anyhow!(
                "unable to create memory buffer from stdin: {}",
                LLVMString::new(err_string)
            ));
        }

        Ok(MemoryBuffer::new(buffer))
    }

    /// Creates a MemoryBuffer with the contents read from stdin
    pub fn create_from_slice<'s>(input: &'s [u8], name: &str) -> MemoryBuffer<'s> {
        use llvm_sys::core::LLVMCreateMemoryBufferWithMemoryRange;

        let name = CString::new(name).unwrap();

        let buffer = unsafe {
            LLVMCreateMemoryBufferWithMemoryRange(
                input.as_ptr() as *const libc::c_char,
                input.len(),
                name.as_ptr(),
                /* requiresNullTerminator= */ false as libc::c_int,
            )
        };

        MemoryBuffer::new(buffer)
    }

    /// Gets a byte slice of this `MemoryBuffer`.
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            let start = LLVMGetBufferStart(self.buffer);
            slice::from_raw_parts(start as *const _, self.get_size())
        }
    }

    /// Gets the byte size of this `MemoryBuffer`.
    pub fn get_size(&self) -> usize {
        unsafe { LLVMGetBufferSize(self.buffer) }
    }

    /// Copies the buffer to the given writer.
    #[allow(unused)]
    pub fn copy_to<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.as_slice())
    }
}

impl<'a> AsRef<llvm_sys::LLVMMemoryBuffer> for MemoryBuffer<'a> {
    fn as_ref(&self) -> &llvm_sys::LLVMMemoryBuffer {
        unsafe { mem::transmute::<MemoryBufferRef, &'a llvm_sys::LLVMMemoryBuffer>(self.buffer) }
    }
}
impl<'a> AsMut<llvm_sys::LLVMMemoryBuffer> for MemoryBuffer<'a> {
    fn as_mut(&mut self) -> &mut llvm_sys::LLVMMemoryBuffer {
        unsafe {
            mem::transmute::<MemoryBufferRef, &'a mut llvm_sys::LLVMMemoryBuffer>(self.buffer)
        }
    }
}

impl<'a> Clone for MemoryBuffer<'a> {
    fn clone(&self) -> Self {
        use llvm_sys::core::LLVMCreateMemoryBufferWithMemoryRangeCopy;

        let name = CString::new("cloned").unwrap();
        let slice = self.as_slice();
        let buffer = unsafe {
            LLVMCreateMemoryBufferWithMemoryRangeCopy(
                slice.as_ptr() as *const libc::c_char,
                slice.len(),
                name.as_ptr(),
            )
        };

        MemoryBuffer::new(buffer)
    }
}

impl<'a> Drop for MemoryBuffer<'a> {
    fn drop(&mut self) {
        unsafe { LLVMDisposeMemoryBuffer(self.buffer) }
    }
}

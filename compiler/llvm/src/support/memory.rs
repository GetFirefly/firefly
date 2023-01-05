use std::borrow::{Borrow, ToOwned};
use std::fmt;
use std::io;
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;

use anyhow::anyhow;

use super::*;

extern "C" {
    type LlvmMemoryBuffer;
}

/// Represents a raw reference to an LLVM memory buffer, it is ownership agnostic
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Buffer(*const LlvmMemoryBuffer);
impl Buffer {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn is_empty(&self) -> bool {
        self.is_null() || self.len() == 0
    }

    /// Returns a pointer to the start of the underlying buffer
    pub fn as_ptr(&self) -> *const u8 {
        extern "C" {
            fn LLVMGetBufferStart(buf: Buffer) -> *const u8;
        }
        unsafe { LLVMGetBufferStart(*self) }
    }

    /// Returns the size of the buffer in bytes
    pub fn len(&self) -> usize {
        extern "C" {
            fn LLVMGetBufferSize(buf: Buffer) -> usize;
        }
        unsafe { LLVMGetBufferSize(*self) }
    }

    /// Gets a byte slice of this `MemoryBuffer`.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Copies the buffer to the given writer.
    pub fn copy_to<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(self.as_slice())
    }
}
impl fmt::Pointer for Buffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}

/// Represents a memory buffer which borrows from some other data, so its lifetime is bound to that
/// data
#[repr(transparent)]
pub struct MemoryBuffer<'a> {
    buffer: Buffer,
    _marker: core::marker::PhantomData<&'a [u8]>,
}
impl<'a> fmt::Pointer for MemoryBuffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.buffer)
    }
}
impl<'a> fmt::Debug for MemoryBuffer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.buffer.is_null() {
            write!(f, "MemoryBuffer(null)")
        } else {
            let len = self.len();
            write!(f, "MemoryBuffer(data = {:p}, len = {})", self.buffer, len)
        }
    }
}
impl<'a> MemoryBuffer<'a> {
    /// Creates a MemoryBuffer that borrows from the given slice
    pub fn create_from_slice<S: Into<StringRef>>(input: &'a [u8], name: S) -> Self {
        extern "C" {
            fn LLVMCreateMemoryBufferWithMemoryRange(
                data: *const u8,
                len: usize,
                name: *const std::os::raw::c_char,
                needs_null_terminator: bool,
            ) -> Buffer;
        }

        let name = name.into();
        let c_str = name.to_cstr();

        let buffer = unsafe {
            LLVMCreateMemoryBufferWithMemoryRange(
                input.as_ptr(),
                input.len(),
                c_str.as_ptr(),
                /* requiresNullTerminator= */ false,
            )
        };
        Self {
            buffer,
            _marker: core::marker::PhantomData,
        }
    }
}
impl<'a> Deref for MemoryBuffer<'a> {
    type Target = Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
impl<'a> ToOwned for MemoryBuffer<'a> {
    type Owned = OwnedMemoryBuffer;

    fn to_owned(&self) -> Self::Owned {
        extern "C" {
            fn LLVMCreateMemoryBufferWithMemoryRangeCopy(
                input: *const u8,
                len: usize,
                name: *const std::os::raw::c_char,
            ) -> Buffer;
        }
        let name = std::ptr::null();
        let buffer =
            unsafe { LLVMCreateMemoryBufferWithMemoryRangeCopy(self.as_ptr(), self.len(), name) };

        OwnedMemoryBuffer(buffer)
    }
}

/// Represents an owned memory buffer
///
/// Creating one of these requires reading a file into memory, or copying data from some other
/// source.
///
/// When an owned memory buffer is dropped, its underlying memory is freed
#[repr(transparent)]
pub struct OwnedMemoryBuffer(Buffer);
impl OwnedMemoryBuffer {
    /// Consumes this wrapper and returns the raw `LLVMMemoryBufferRef`
    ///
    /// This is used when handing ownership of the buffer to a function in LLVM,
    /// requiring us to intentionally "leak" the `MemoryBuffer` wrapper so that
    /// the drop implementation is not run.
    pub fn release(self) -> MemoryBuffer<'static> {
        let buffer = self.0;
        std::mem::forget(self);
        MemoryBuffer {
            buffer,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates a MemoryBuffer with the contents of the file at the given path
    pub fn create_from_file<P: AsRef<Path>>(file: P) -> anyhow::Result<OwnedMemoryBuffer> {
        extern "C" {
            fn LLVMCreateMemoryBufferWithContentsOfFile(
                path: *const std::os::raw::c_char,
                buf: *mut Buffer,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let file = file.as_ref();
        if !file.exists() || !file.is_file() {
            return Err(anyhow!(
                "unable to create memory buffer from {}: not a file",
                file.to_string_lossy()
            ));
        }
        let file = StringRef::from(file);
        let path = file.to_cstr();

        let mut buffer = MaybeUninit::uninit();
        let mut error = MaybeUninit::uninit();

        let failed = unsafe {
            LLVMCreateMemoryBufferWithContentsOfFile(
                path.as_ptr(),
                buffer.as_mut_ptr(),
                error.as_mut_ptr(),
            )
        };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!(
                "unable to create memory buffer from file: {}",
                &error
            ))
        } else {
            Ok(Self(unsafe { buffer.assume_init() }))
        }
    }

    /// Creates a MemoryBuffer with the contents read from stdin
    pub fn create_from_stdin() -> anyhow::Result<OwnedMemoryBuffer> {
        extern "C" {
            fn LLVMCreateMemoryBufferWithSTDIN(
                buf: *mut Buffer,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let mut buffer = MaybeUninit::uninit();
        let mut error = MaybeUninit::uninit();

        let failed =
            unsafe { LLVMCreateMemoryBufferWithSTDIN(buffer.as_mut_ptr(), error.as_mut_ptr()) };

        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!(
                "unable to create memory buffer from stdin: {}",
                &error
            ))
        } else {
            Ok(Self(unsafe { buffer.assume_init() }))
        }
    }

    pub fn borrow<'a>(&'a self) -> MemoryBuffer<'a> {
        MemoryBuffer {
            buffer: self.0,
            _marker: core::marker::PhantomData,
        }
    }
}
impl Deref for OwnedMemoryBuffer {
    type Target = Buffer;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<'a> Borrow<MemoryBuffer<'a>> for OwnedMemoryBuffer {
    fn borrow(&self) -> &MemoryBuffer<'a> {
        unsafe { std::mem::transmute::<&OwnedMemoryBuffer, &MemoryBuffer<'a>>(self) }
    }
}
impl fmt::Pointer for OwnedMemoryBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl fmt::Debug for OwnedMemoryBuffer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "OwnedMemoryBuffer(data = {:p}, len = {})",
            self.0,
            self.len()
        )
    }
}
impl Drop for OwnedMemoryBuffer {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeMemoryBuffer(buf: Buffer);
        }
        unsafe { LLVMDisposeMemoryBuffer(self.0) }
    }
}

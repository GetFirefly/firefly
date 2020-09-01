use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::CStr;
use std::fmt;
use std::ops::Deref;
use std::slice;
use std::string::FromUtf8Error;

extern "C" {
    pub type Twine;
}

pub fn build_string<'a>(f: impl FnOnce(&RustString)) -> Option<String> {
    let rs = RustString {
        bytes: RefCell::new(Vec::new()),
    };
    f(&rs);
    let bytes = rs.bytes.into_inner();
    if bytes.len() > 0 {
        Some(String::from_utf8_lossy(&bytes).into_owned())
    } else {
        None
    }
}

pub fn twine_to_string(twine: &Twine) -> Option<String> {
    build_string(|s| unsafe { LLVMLumenWriteTwineToString(twine, s) })
}

#[repr(C)]
pub struct RustString {
    bytes: RefCell<Vec<u8>>,
}

/// Appending to a Rust string -- used by RawRustStringOstream
#[no_mangle]
#[allow(improper_ctypes)]
pub unsafe extern "C" fn LLVMRustStringWriteImpl(
    rs: &RustString,
    ptr: *const libc::c_char,
    size: libc::size_t,
) {
    let slice = slice::from_raw_parts(ptr as *const u8, size as usize);
    rs.bytes.borrow_mut().extend_from_slice(slice);
}

#[derive(Eq)]
pub struct LLVMString(*const libc::c_char);
impl LLVMString {
    pub fn new(ptr: *const libc::c_char) -> Self {
        Self(ptr)
    }

    pub fn create(string: &str) -> LLVMString {
        use crate::sys::core::LLVMCreateMessage;

        debug_assert_eq!(string.as_bytes()[string.as_bytes().len() - 1], 0);
        let ptr = unsafe { LLVMCreateMessage(string.as_ptr() as *const _) };
        LLVMString::new(ptr)
    }

    pub fn to_string(&self) -> String {
        (*self).to_string_lossy().into_owned()
    }
}
impl Deref for LLVMString {
    type Target = CStr;

    fn deref(&self) -> &Self::Target {
        unsafe { CStr::from_ptr(self.0) }
    }
}
impl fmt::Debug for LLVMString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}
impl fmt::Display for LLVMString {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.deref())
    }
}
impl PartialEq for LLVMString {
    fn eq(&self, other: &LLVMString) -> bool {
        **self == **other
    }
}
impl std::error::Error for LLVMString {}
impl Drop for LLVMString {
    fn drop(&mut self) {
        use crate::sys::core::LLVMDisposeMessage;
        unsafe {
            LLVMDisposeMessage(self.0 as *mut _);
        }
    }
}

extern "C" {
    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteTwineToString(T: &Twine, Str: &RustString);
}

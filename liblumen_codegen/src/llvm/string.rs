use std::ffi::CStr;
use std::fmt;
use std::ops::Deref;

#[derive(Eq)]
pub struct LLVMString(*const libc::c_char);
impl LLVMString {
    pub fn new(ptr: *const libc::c_char) -> Self {
        Self(ptr)
    }

    #[allow(unused)]
    pub fn create(string: &str) -> LLVMString {
        use llvm_sys::core::LLVMCreateMessage;

        debug_assert_eq!(string.as_bytes()[string.as_bytes().len() - 1], 0);
        let ptr = unsafe { LLVMCreateMessage(string.as_ptr() as *const _) };
        LLVMString::new(ptr)
    }

    #[allow(unused)]
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
        use llvm_sys::core::LLVMDisposeMessage;
        unsafe {
            LLVMDisposeMessage(self.0 as *mut _);
        }
    }
}

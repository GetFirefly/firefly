use std::borrow::{Borrow, Cow};
use std::ffi::{c_void, CStr, CString, OsStr};
use std::fmt::{self, Display};
use std::ops::Deref;
use std::path::{Path, PathBuf};

use liblumen_intern::Symbol;

/// OwnedStringRef's represent a null-terminated string allocated by LLVM
/// over which ownership is our responsibility. In all respects these behave
/// like a regular StringRef, with the exception that they are always null
/// terminated.
#[repr(transparent)]
pub struct OwnedStringRef(StringRef);
impl OwnedStringRef {
    /// Converts this to a CStr reference
    pub fn as_cstr<'a>(&'a self) -> &'a CStr {
        assert!(!self.data.is_null());
        unsafe { CStr::from_ptr(self.0.data as *const std::os::raw::c_char) }
    }

    /// Converts a pointer to a null-terminated C string into an OwnedStringRef
    ///
    /// This is highly unsafe, and should be used with APIs that return `const char*` with no length
    pub unsafe fn from_ptr(ptr: *const std::os::raw::c_char) -> Self {
        if ptr.is_null() {
            return Self(StringRef::default());
        }
        let c_str = CStr::from_ptr(ptr);
        Self(c_str.into())
    }
}
impl From<String> for OwnedStringRef {
    fn from(s: String) -> Self {
        extern "C" {
            fn LLVMCreateMessage(ptr: *const std::os::raw::c_char) -> *const std::os::raw::c_char;
        }
        let c_str = CString::new(s).unwrap();
        unsafe { Self::from_ptr(LLVMCreateMessage(c_str.as_ptr())) }
    }
}
impl Into<String> for OwnedStringRef {
    fn into(self) -> String {
        self.0.into()
    }
}
impl Borrow<StringRef> for OwnedStringRef {
    fn borrow(&self) -> &StringRef {
        &self.0
    }
}
impl Deref for OwnedStringRef {
    type Target = StringRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedStringRef {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeMessage(message: *const u8);
        }
        if self.0.is_null() {
            return;
        }
        unsafe { LLVMDisposeMessage(self.0.data) }
    }
}
impl Display for OwnedStringRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Corresponds to `llvm::StringRef`, which as you'd guess, is a
/// string reference - a pointer to a byte vector with a length.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct StringRef {
    pub data: *const u8,
    pub len: usize,
}
impl StringRef {
    /// Returns true if this string ref has a null data pointer
    #[inline]
    pub fn is_null(&self) -> bool {
        self.data.is_null()
    }

    /// Returns true if this string ref has no data
    #[inline]
    pub fn empty(&self) -> bool {
        self.data.is_null() || self.len == 0
    }

    /// You should prefer to use `From`/`Into` or `TryFrom`/`TryInto` instead of this
    pub unsafe fn from_raw_parts(data: *const u8, len: usize) -> Self {
        Self { data, len }
    }

    /// This function is used to construct a StringRef from a null-terminated C-style string
    ///
    /// This is equivalent to `CStr::from_ptr`, but allows you to avoid the extra song and dance
    /// in places where you are frequently working with these types of strings.
    ///
    /// NOTE: This is entirely unsafe, if you pass a pointer to a string that is not null-terminated,
    /// it _will_ go wrong, so make sure you know what you're doing.
    pub unsafe fn from_ptr(s: *const std::os::raw::c_char) -> Self {
        CStr::from_ptr(s).into()
    }

    /// Returns true if this string is null terminated
    ///
    /// NOTE: This is linear in the size of the underlying memory, you
    /// should only use this when it is important to distinguish whether
    /// or not null termination is required for the string
    pub fn is_null_terminated(&self) -> bool {
        assert!(!self.data.is_null());
        let bytes = self.as_bytes();
        bytes.iter().copied().any(|b| b == 0)
    }

    /// Return the underlying data as a byte slice
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        use core::slice;
        unsafe { slice::from_raw_parts(self.data, self.len) }
    }

    /// Converts this stringref to a CStr/CString depending on whether it is null-terminated or not
    pub fn to_cstr(&self) -> Cow<'_, CStr> {
        if self.is_null_terminated() {
            Cow::Borrowed(unsafe { CStr::from_ptr(self.data as *const std::os::raw::c_char) })
        } else {
            Cow::Owned(CString::new(self.as_bytes()).unwrap())
        }
    }

    /// Converts this stringref to a Path/PathBuf depending on whether it is a lossy conversion
    pub fn to_path_lossy(&self) -> Cow<'_, Path> {
        match String::from_utf8_lossy(self.as_bytes()) {
            Cow::Owned(s) => Cow::Owned(PathBuf::from(s)),
            Cow::Borrowed(s) => Cow::Borrowed(Path::new(s)),
        }
    }
}
impl Default for StringRef {
    fn default() -> StringRef {
        Self {
            data: core::ptr::null(),
            len: 0,
        }
    }
}
impl From<Symbol> for StringRef {
    #[inline]
    fn from(s: Symbol) -> Self {
        let bytes = s.as_str().get().as_bytes();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&str> for StringRef {
    #[inline]
    fn from(s: &str) -> Self {
        let bytes = s.as_bytes();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&CStr> for StringRef {
    fn from(s: &CStr) -> Self {
        let bytes = s.to_bytes_with_nul();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&CString> for StringRef {
    fn from(s: &CString) -> Self {
        let bytes = s.to_bytes_with_nul();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&OsStr> for StringRef {
    #[cfg(all(unix, target_env = "wasi"))]
    fn from(s: &OsStr) -> Self {
        use std::os::wasi::ffi::OsStrExt;
        let bytes = s.as_bytes();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    #[cfg(all(unix, not(target_env = "wasi")))]
    fn from(s: &OsStr) -> Self {
        use std::os::unix::ffi::OsStrExt;
        let bytes = s.as_bytes();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    #[cfg(windows)]
    fn from(s: &OsStr) -> Self {
        use std::os::windows::ffi::OsStrExt;
        s.to_str().into()
    }
}
impl From<&String> for StringRef {
    #[inline]
    fn from(s: &String) -> Self {
        let bytes = s.as_bytes();
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&[u8]> for StringRef {
    #[inline(always)]
    fn from(bytes: &[u8]) -> Self {
        Self {
            data: bytes.as_ptr(),
            len: bytes.len(),
        }
    }
}
impl From<&Path> for StringRef {
    #[inline]
    #[cfg(all(target_arch = "wasm32", target_os = "wasi"))]
    fn from(path: &Path) -> Self {
        use std::os::wasi::ffi::OsStrExt;
        path.as_os_str().as_bytes().into()
    }
    #[cfg(all(unix, not(target_os = "wasi")))]
    fn from(path: &Path) -> Self {
        use std::os::unix::ffi::OsStrExt;
        path.as_os_str().as_bytes().into()
    }
    #[cfg(windows)]
    fn from(path: &Path) -> Self {
        path.to_str().unwrap().as_bytes().into()
    }
}
impl<'a> TryInto<&'a str> for StringRef {
    type Error = core::str::Utf8Error;

    #[inline]
    fn try_into(self) -> Result<&'a str, Self::Error> {
        use core::slice;
        let bytes = unsafe { slice::from_raw_parts(self.data, self.len) };

        std::str::from_utf8(&bytes)
    }
}
impl<'a> TryInto<&'a Path> for StringRef {
    type Error = core::str::Utf8Error;

    #[inline]
    fn try_into(self) -> Result<&'a Path, Self::Error> {
        use core::slice;
        let bytes = unsafe { slice::from_raw_parts(self.data, self.len) };

        std::str::from_utf8(&bytes).map(|s| Path::new(s))
    }
}
impl<'a> Into<&'a [u8]> for StringRef {
    #[inline]
    fn into(self) -> &'a [u8] {
        use core::slice;
        unsafe { slice::from_raw_parts(self.data, self.len) }
    }
}
impl Into<Vec<u8>> for StringRef {
    #[inline]
    fn into(self) -> Vec<u8> {
        self.as_bytes().to_vec()
    }
}
impl Into<String> for StringRef {
    #[inline]
    fn into(self) -> String {
        use core::slice;
        let bytes = unsafe { slice::from_raw_parts(self.data, self.len) };

        String::from_utf8_lossy(&bytes).into_owned()
    }
}
impl Eq for StringRef {}
impl PartialEq for StringRef {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}
impl PartialEq<str> for StringRef {
    fn eq(&self, other: &str) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}
impl PartialEq<String> for StringRef {
    fn eq(&self, other: &String) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}
impl PartialEq<[u8]> for StringRef {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }
}
impl Display for StringRef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::slice;
        let bytes = unsafe { slice::from_raw_parts(self.data, self.len) };
        let s = match bytes.strip_suffix(&[0]) {
            None => String::from_utf8_lossy(&bytes),
            Some(stripped) => String::from_utf8_lossy(stripped),
        };
        write!(f, "{}", &s)
    }
}

/// A callback for returning string references.
///
/// This function is called back by the functions that need to return a reference to
/// the portion of the string.
///
/// The final parameter is a pointer to user data forwarded from the printing call.
pub type MlirStringCallback = extern "C" fn(data: StringRef, userdata: *mut c_void);

pub extern "C" fn write_to_formatter(data: StringRef, result: *mut c_void) {
    let f = unsafe { &mut *(result as *mut fmt::Formatter) };
    write!(f, "{}", &data).unwrap();
}

use std::cell::RefCell;
use std::slice;
use std::string::FromUtf8Error;

extern "C" {
    pub type Twine;
}

#[repr(C)]
pub struct RustString {
    bytes: RefCell<Vec<u8>>,
}

extern "C" {
    #[allow(improper_ctypes)]
    pub fn LLVMLumenWriteTwineToString(T: &Twine, Str: &RustString);
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

pub fn build_string(f: impl FnOnce(&RustString)) -> Result<String, FromUtf8Error> {
    let rs = RustString {
        bytes: RefCell::new(Vec::new()),
    };
    f(&rs);
    String::from_utf8(rs.bytes.into_inner())
}

pub fn twine_to_string(twine: &Twine) -> String {
    unsafe {
        build_string(|s| LLVMLumenWriteTwineToString(twine, s))
            .expect("got a non-UTF8 Twine from LLVM")
    }
}

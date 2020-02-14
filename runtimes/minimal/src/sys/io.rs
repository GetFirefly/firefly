use std::ffi::CStr;

use libc;

#[link_name = "__io_put_chars_1"]
pub extern "C" fn put_chars_1(s: *const libc::c_char) -> Option<Term> {
    let sref = unsafe { CStr::from_ptr(s).to_string_lossy() };
    println!(&sref);
    Some(ok!())
}

#[link_name = "__io_format_2"]
pub extern "C" fn format_2(
    s: *const libc::c_char,
    argv: *const Term,
    argc: libc::c_uint,
) -> Option<Term> {
    unimplemented!();
}

#[link_name = "__io_nl_0"]
pub extern "C" fn nl_0() -> Option<Term> {
    println!();
    Some(ok!())
}

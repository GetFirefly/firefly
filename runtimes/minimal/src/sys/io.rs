use std::ffi::CStr;

use libc;

use liblumen_alloc::erts::term::prelude::*;

pub use lumen_rt_core::sys::io::puts;

#[export_name = "__lumen_builtin_printf"]
pub extern "C" fn printf_1(term: Term) -> Term {
    match term.decode() {
        Ok(tt) => {
            println!("{}", tt);
            Atom::from_str("ok").encode().unwrap()
        }
        Err(reason) => {
            println!("ERR: {:?}", reason);
            Term::NONE
        }
    }
}

#[export_name = "io:put_chars/1"]
pub extern "C" fn put_chars_1(s: *const libc::c_char) -> Option<Term> {
    let sref = unsafe { CStr::from_ptr(s).to_string_lossy() };
    println!("{}", &sref);
    Some(ok!())
}

#[export_name = "io:format/2"]
pub extern "C" fn format_2(
    _s: *const libc::c_char,
    _argv: *const Term,
    _argc: libc::c_uint,
) -> Option<Term> {
    unimplemented!();
}

#[export_name = "io:nl/0"]
pub extern "C" fn nl_0() -> Option<Term> {
    println!();
    Some(ok!())
}

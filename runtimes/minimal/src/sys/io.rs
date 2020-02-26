use std::ffi::CStr;

use libc;

use liblumen_alloc::erts::term::prelude::*;

#[export_name = "__lumen_builtin_printf"]
pub extern "C" fn printf_1(term: Term) -> Term {
    use liblumen_alloc::erts::term::arch::Repr;
    match term.decode() {
        Ok(TypedTerm::ProcBin(boxed)) => {
            let ptr = boxed.as_ptr();
            println!("{:p}", ptr);
            Atom::from_str("ok").encode().unwrap()
        }
        Ok(_) => {
            println!("ERR: wrong term type");
            Term::NONE
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
    s: *const libc::c_char,
    argv: *const Term,
    argc: libc::c_uint,
) -> Option<Term> {
    unimplemented!();
}

#[export_name = "io:nl/0"]
pub extern "C" fn nl_0() -> Option<Term> {
    println!();
    Some(ok!())
}

pub fn puts(s: &str) {
    println!("{}", s);
}

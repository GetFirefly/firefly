use std::panic;

use liblumen_alloc::erts::term::prelude::*;

/// Strict equality
#[export_name = "__lumen_builtin_cmpeq"]
pub extern "C" fn builtin_cmpeq(lhs: Term, rhs: Term) -> bool {
    let result = panic::catch_unwind(|| {
        if let Ok(left) = lhs.decode() {
            if let Ok(right) = rhs.decode() {
                left.exact_eq(&right)
            } else {
                //Atom::str_to_term("false")
                false
            }
        } else {
            if lhs.is_none() && rhs.is_none() {
                //Atom::str_to_term("true")
                true
            } else {
                //Atom::str_to_term("false")
                false
            }
        }
    });
    if let Ok(res) = result {
        res
    } else {
        false
        //Atom::str_to_term("false")
    }
}

/// Capture the current stack trace
#[export_name = "__lumen_builtin_trace_capture"]
pub extern "C" fn builtin_trace_capture() -> Term {
    // TODO:
    Term::NIL
}

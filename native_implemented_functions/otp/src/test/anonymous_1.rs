use std::ffi::c_void;

use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::*;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

pub fn anonymous_closure(process: &Process) -> AllocResult<Term> {
    process.anonymous_closure_with_env_from_slice(
        super::module(),
        INDEX,
        OLD_UNIQUE,
        UNIQUE,
        ARITY,
        Some(native as *const c_void),
        process.pid().into(),
        &[],
    )
}

const INDEX: Index = 1;
const OLD_UNIQUE: OldUnique = 2;
const UNIQUE: Unique = [
    0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF, 0x01,
];

#[native_implemented_function(1-2-23456789ABCDEF0123456789ABCDEF01/1)]
fn result(argument: Term) -> Term {
    argument
}

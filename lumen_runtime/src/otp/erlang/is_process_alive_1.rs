// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::registry::pid_to_process;
use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(is_process_alive/1)]
pub fn native(process: &Process, term: Term) -> exception::Result<Term> {
    if term == process.pid_term() {
        Ok((!process.is_exiting()).into())
    } else {
        let pid: Pid = term.try_into()?;

        match pid_to_process(&pid) {
            Some(arc_process) => Ok((!arc_process.is_exiting()).into()),
            None => Ok(false.into()),
        }
    }
}

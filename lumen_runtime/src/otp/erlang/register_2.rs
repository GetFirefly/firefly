// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

use crate::registry;

#[native_implemented_function(register/2)]
pub fn native(arc_process: Arc<Process>, name: Term, pid_or_port: Term) -> exception::Result {
    let atom: Atom = name.try_into()?;

    let option_registered: Option<Term> = match atom.name() {
        "undefined" => None,
        _ => match pid_or_port.to_typed_term().unwrap() {
            TypedTerm::Pid(pid) => {
                registry::pid_to_self_or_process(pid, &arc_process).and_then(|pid_arc_process| {
                    if registry::put_atom_to_process(atom, pid_arc_process) {
                        Some(true.into())
                    } else {
                        None
                    }
                })
            }
            _ => None,
        },
    };

    match option_registered {
        Some(registered) => Ok(registered),
        None => Err(badarg!().into()),
    }
}

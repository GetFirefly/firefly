// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry;

#[native_implemented_function(register/2)]
pub fn result(arc_process: Arc<Process>, name: Term, pid_or_port: Term) -> exception::Result<Term> {
    let atom = term_try_into_atom!(name)?;

    match atom.name() {
        "undefined" => Err(anyhow!("undefined is not an allowed registered name").into()),
        _ => {
            match pid_or_port.decode()? {
                TypedTerm::Pid(pid) => {
                    match registry::pid_to_self_or_process(pid, &arc_process) {
                        Some(pid_arc_process) => {
                            if registry::put_atom_to_process(atom, pid_arc_process) {
                                Ok(true.into())
                            } else {
                                Err(anyhow!("{} could not be registered as {}.  It may already be registered.", pid, atom).into())
                            }
                        }
                        None => Err(anyhow!("{} is not a pid of an alive process", pid).into()),
                    }
                }
                TypedTerm::ExternalPid(_) => Err(anyhow!(
                    "{} is an external pid, but only local pids can be registered",
                    pid_or_port
                )
                .into()),
                TypedTerm::Port(_) => unimplemented!(),
                TypedTerm::ExternalPort(_) => Err(anyhow!(
                    "{} is an external port, but only local ports can be registered",
                    pid_or_port
                )
                .into()),
                _ => Err(anyhow!("{} must be a local pid or port", pid_or_port).into()),
            }
        }
    }
}

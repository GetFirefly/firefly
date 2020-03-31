// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::error;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry::pid_to_process;

#[native_implemented_function(link/1)]
fn native(process: &Process, pid_or_port: Term) -> exception::Result<Term> {
    match pid_or_port.decode()? {
        TypedTerm::Pid(pid) => {
            if pid == process.pid() {
                Ok(true.into())
            } else {
                match pid_to_process(&pid) {
                    Some(pid_arc_process) => {
                        process.link(&pid_arc_process);

                        Ok(true.into())
                    }
                    None => Err(error!(
                        Atom::str_to_term("noproc"),
                        anyhow!("pid ({}) doesn't refer to an alive local process", pid).into()
                    )
                    .into()),
                }
            }
        }
        TypedTerm::Port(_) => unimplemented!(),
        TypedTerm::ExternalPid(_) => unimplemented!(),
        TypedTerm::ExternalPort(_) => unimplemented!(),
        _ => Err(TypeError)
            .context(format!(
                "pid_or_port ({}) is neither a pid nor a port",
                pid_or_port
            ))
            .map_err(From::from),
    }
}

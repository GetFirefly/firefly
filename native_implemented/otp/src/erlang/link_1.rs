#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, error};
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::registry::pid_to_process;

#[native_implemented::function(erlang:link/1)]
fn result(process: &Process, pid_or_port: Term) -> exception::Result<Term> {
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
                    None => Err(error(
                        Atom::str_to_term("noproc"),
                        None,
                        Trace::capture(),
                        Some(
                            anyhow!("pid ({}) doesn't refer to an alive local process", pid).into(),
                        ),
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

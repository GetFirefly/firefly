#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::registry::pid_to_process;

#[native_implemented::function(erlang:unlink/1)]
fn result(process: &Process, pid_or_port: Term) -> Result<Term, NonNull<ErlangException>> {
    match pid_or_port {
        Term::Pid(pid) => {
            if pid == process.pid() {
                Ok(true.into())
            } else {
                match pid_to_process(&pid) {
                    Some(pid_arc_process) => {
                        process.unlink(&pid_arc_process);
                    }
                    None => (),
                }

                Ok(true.into())
            }
        }
        Term::Port(_) => unimplemented!(),
        _ => Err(TypeError)
            .context(format!(
                "pid_or_port ({}) is neither a pid nor a port",
                pid_or_port
            ))
            .map_err(From::from),
    }
}

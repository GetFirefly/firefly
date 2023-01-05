#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;
use std::sync::Arc;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::registry;

#[native_implemented::function(erlang:register/2)]
pub fn result(
    arc_process: Arc<Process>,
    name: Term,
    pid_or_port: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let atom = term_try_into_atom!(name)?;

    match atom.as_str() {
        "undefined" => Err(anyhow!("undefined is not an allowed registered name").into()),
        _ => {
            match pid_or_port {
                Term::Pid(pid) => {
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
                Term::Port(_) => unimplemented!(),
                _ => Err(anyhow!("{} must be a local pid or port", pid_or_port).into()),
            }
        }
    }
}

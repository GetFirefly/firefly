// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::registry::pid_to_process;

macro_rules! is_not_alive {
    ($name:ident) => {
        is_not_alive(stringify!($name), $name)
    };
}

#[native_implemented_function(group_leader/2)]
pub fn result(process: &Process, group_leader: Term, pid: Term) -> exception::Result<Term> {
    let group_leader_pid: Pid = term_try_into_local_pid!(group_leader)?;

    if (group_leader_pid == process.pid()) || pid_to_process(&group_leader_pid).is_some() {
        let pid_pid: Pid = term_try_into_local_pid!(pid)?;

        if process.pid() == pid_pid {
            process.set_group_leader_pid(group_leader_pid);

            Ok(true.into())
        } else {
            match pid_to_process(&pid_pid) {
                Some(pid_arc_process) => {
                    pid_arc_process.set_group_leader_pid(group_leader_pid);

                    Ok(true.into())
                }
                None => is_not_alive!(pid),
            }
        }
    } else {
        is_not_alive!(group_leader)
    }
}

fn is_not_alive(name: &'static str, value: Term) -> exception::Result<Term> {
    Err(anyhow!("{} ({}) is not alive", name, value)).map_err(From::from)
}

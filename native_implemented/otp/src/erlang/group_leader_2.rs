#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::{Pid, Term};

use crate::runtime::registry::pid_to_process;

macro_rules! is_not_alive {
    ($name:ident) => {
        is_not_alive(stringify!($name), $name)
    };
}

#[native_implemented::function(erlang:group_leader/2)]
pub fn result(
    process: &Process,
    group_leader: Term,
    pid: Term,
) -> Result<Term, NonNull<ErlangException>> {
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

fn is_not_alive(name: &'static str, value: Term) -> Result<Term, NonNull<ErlangException>> {
    Err(anyhow!("{} ({}) is not alive", name, value)).map_err(From::from)
}

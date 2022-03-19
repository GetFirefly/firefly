pub mod exceptions;
pub mod gc;
pub mod receive;

use std::panic;

use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::process::current_process;
use lumen_rt_core::registry;

#[export_name = "erlang:!/2"]
pub extern "C-unwind" fn builtin_send(to_term: Term, msg: Term) -> Term {
    let result = panic::catch_unwind(|| {
        let decoded_result: Result<Pid, _> = to_term.decode().unwrap().try_into();
        if let Ok(to) = decoded_result {
            let p = current_process();
            let self_pid = p.pid();
            if self_pid == to {
                p.send_from_self(msg);
                return msg;
            } else {
                if let Some(ref to_proc) = registry::pid_to_process(&to) {
                    to_proc.send_from_other(msg);
                    crate::scheduler::stop_waiting(to_proc);
                }

                return msg;
            }
        } else {
            // TODO: badarg
            panic!("invalid pid: {:?}", to_term);
        }
    });
    if let Ok(res) = result {
        res
    } else {
        panic!("send failed");
    }
}

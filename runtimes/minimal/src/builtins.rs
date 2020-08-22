pub mod gc;
pub mod receive;

use std::convert::TryInto;
use std::panic;

use anyhow::anyhow;

use liblumen_alloc::erts::process::ffi::process_raise;
use liblumen_alloc::erts::term::prelude::*;

use lumen_rt_core::process::current_process;
use lumen_rt_core::registry;

#[export_name = "erlang:!/2"]
pub extern "C" fn builtin_send(to_term: Term, msg: Term) -> Term {
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
                    if let Ok(resume) = to_proc.send_from_other(msg) {
                        if resume {
                            crate::scheduler::stop_waiting(to_proc);
                        }
                        return msg;
                    } else {
                        panic!("error during send");
                    }
                } else {
                    return msg;
                }
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

#[export_name = "__lumen_builtin_fail/1"]
pub extern "C" fn builtin_fail(reason: Term) -> Term {
    use liblumen_alloc::erts::exception::{self, ArcError, RuntimeException};
    if reason.is_none() {
        reason
    } else {
        let err = RuntimeException::Error(exception::Error::new(
            reason,
            None,
            ArcError::new(anyhow!("call to fail/1 raised an error!")),
        ));
        process_raise(&current_process(), err);
    }
}

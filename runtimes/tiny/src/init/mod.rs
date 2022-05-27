use liblumen_rt::function::ErlangResult;
use liblumen_rt::process::Process;
use liblumen_rt::term::{ListBuilder, OpaqueTerm};

use crate::env;
use crate::scheduler;

#[export_name = "init:get_plain_arguments/0"]
pub extern "C-unwind" fn get_plain_arguments() -> ErlangResult {
    scheduler::with_current(|scheduler| {
        get_plain_arguments_with_process(&scheduler.current_process())
    })
}

fn get_plain_arguments_with_process(process: &Process) -> ErlangResult {
    let argv = env::get_argv();
    if argv.is_empty() {
        return Ok(OpaqueTerm::NIL);
    }

    let mut builder = ListBuilder::new(process);
    for arg in argv.iter().copied() {
        // TODO: Properly handle allocation failure
        builder.push(arg.into()).unwrap();
    }

    match builder.finish() {
        Some(ptr) => Ok(ptr.into()),
        None => Ok(OpaqueTerm::NIL),
    }
}

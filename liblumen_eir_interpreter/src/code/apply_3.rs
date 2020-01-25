use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::Arity;

use locate_code::locate_code;

use super::interpreter_mfa;

#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let module_term = arc_process.stack_pop().unwrap();
    let function_term = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let module: Atom = module_term.try_into().unwrap();
    let function: Atom = function_term.try_into().unwrap();

    let arity: Arity = match argument_list.decode().unwrap() {
        TypedTerm::Nil => panic!(),
        TypedTerm::List(argument_cons) => {
            (argument_cons.into_iter().count() - 2).try_into().unwrap()
        }
        _ => panic!(),
    };

    let frame = Frame::new(
        module,
        function,
        arity,
        interpreter_mfa::LOCATION,
        interpreter_mfa::code,
    );

    arc_process.stack_push(argument_list)?;
    arc_process.replace_frame(frame);

    Process::call_code(arc_process)
}

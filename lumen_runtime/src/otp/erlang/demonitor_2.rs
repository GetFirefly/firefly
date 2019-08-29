mod options;

// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;

use liblumen_alloc::erts::term::{Atom, Boxed, Reference, Term};
use liblumen_alloc::ModuleFunctionArity;

use crate::otp::erlang::demonitor_2::options::Options;
use crate::process::monitor::is_down;
use crate::registry::pid_to_process;

pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
    placement: Placement,
    reference: Term,
    options: Term,
) -> Result<(), Alloc> {
    process.stack_push(options)?;
    process.stack_push(reference)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let reference = arc_process.stack_pop().unwrap();
    let options = arc_process.stack_pop().unwrap();

    match native(arc_process, reference, options) {
        Ok(true_term) => {
            arc_process.return_from_call(true_term)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn demonitor(
    monitoring_process: &ProcessControlBlock,
    reference: &Reference,
    Options { flush, info }: Options,
) -> exception::Result {
    match monitoring_process.demonitor(reference) {
        Some(monitored_pid) => {
            match pid_to_process(&monitored_pid) {
                Some(monitored_arc_proces) => match monitored_arc_proces.demonitored(reference) {
                    Some(monitoring_pid) => assert_eq!(monitoring_process.pid(), monitoring_pid),
                    None => (),
                },
                None => (),
            }

            if flush {
                let flushed = self::flush(monitoring_process, reference);

                if info && flushed {
                    Ok(false.into())
                } else {
                    Ok(true.into())
                }
            } else {
                Ok(true.into())
            }
        }
        None => {
            if info {
                Ok(false.into())
            } else {
                Ok(true.into())
            }
        }
    }
}

fn flush(monitoring_process: &ProcessControlBlock, reference: &Reference) -> bool {
    monitoring_process
        .mailbox
        .lock()
        .borrow_mut()
        .flush(|message| is_down(message, reference), monitoring_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("demonitor").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub fn native(process: &ProcessControlBlock, reference: Term, options: Term) -> exception::Result {
    let reference_reference: Boxed<Reference> = reference.try_into()?;
    let options_options: Options = options.try_into()?;

    demonitor(process, &reference_reference, options_options)
}

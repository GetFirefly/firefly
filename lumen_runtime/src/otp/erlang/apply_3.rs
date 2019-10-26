#[cfg(test)]
use std::convert::TryInto;
use std::sync::Arc;

use liblumen_core::locks::RwLock;

#[cfg(test)]
use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
#[cfg(not(test))]
use liblumen_alloc::erts::exception::Exception;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
#[cfg(test)]
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::ModuleFunctionArity;

#[cfg(test)]
use crate::otp::erlang;

/// Returns the `Code` that should be used in `otp::erlang::spawn_3` to look up and call a known
/// BIF or user function using the MFA.
///
/// ## Preconditons
///
/// ### Stack
///
/// 1. module - atom `Term`
/// 2. function - atom `Term`
/// 3. arguments - list `Term`
///
/// ## Post-conditions
///
/// ### Ok
///
/// #### Stack
///
/// 1. return - Term
///
/// ### Err
///
/// #### Process
///
/// * `status` - `Status::Exiting` with exception from lookup or called function.
pub fn get_code() -> Code {
    *RW_LOCK_CODE.read()
}

pub fn set_code(code: Code) {
    *RW_LOCK_CODE.write() = code;
}

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    module: Term,
    function: Term,
    arguments: Term,
) -> Result<(), Alloc> {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

/// Treats all MFAs as undefined.
#[cfg(not(test))]
fn code(arc_process: &Arc<Process>) -> code::Result {
    // arguments are consumed, but unused
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();
    arc_process.reduce();

    match liblumen_alloc::undef!(arc_process, module, function, arguments) {
        Exception::Runtime(runtime_exception) => {
            arc_process.exception(runtime_exception);

            Ok(())
        }
        Exception::System(system_exception) => Err(system_exception),
    }
}

#[cfg(test)]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();

    match argument_list.decode().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!("{:?} is not an argument list", argument_list),
    }

    let arity = argument_vec.len();

    let module_atom: Atom = module.try_into().unwrap();
    let function_atom: Atom = function.try_into().unwrap();

    match module_atom.name() {
        "erlang" => match function_atom.name() {
            "+" => match arity {
                1 => {
                    erlang::number_or_badarith_1::place_frame_with_arguments(
                        arc_process,
                        Placement::Replace,
                        argument_vec[0],
                    )?;

                    Process::call_code(arc_process)
                }
                _ => undef(arc_process, module, function, argument_list),
            },
            "self" => match arity {
                0 => {
                    erlang::self_0::place_frame_with_arguments(arc_process, Placement::Replace)?;

                    Process::call_code(arc_process)
                }
                _ => undef(arc_process, module, function, argument_list),
            },
            _ => undef(arc_process, module, function, argument_list),
        },
        _ => undef(arc_process, module, function, argument_list),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), get_code())
}

fn function() -> Atom {
    Atom::try_from_str("apply").unwrap()
}

pub(crate) fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    })
}

#[cfg(test)]
fn undef(
    arc_process: &Arc<Process>,
    module: Term,
    function: Term,
    arguments: Term,
) -> code::Result {
    arc_process.reduce();
    let exception = liblumen_alloc::undef!(arc_process, module, function, arguments);
    let runtime_exception: runtime::Exception = exception.try_into().unwrap();
    arc_process.exception(runtime_exception);

    Ok(())
}

lazy_static! {
    static ref RW_LOCK_CODE: RwLock<Code> = RwLock::new(code);
}

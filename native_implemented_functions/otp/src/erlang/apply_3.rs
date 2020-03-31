use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;
use lazy_static::lazy_static;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::{self, ArcError};
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

const ARITY: Arity = 3;

pub fn export() {
    crate::runtime::code::export::insert(super::module(), function(), ARITY, get_code());
}

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
) -> code::Result {
    process.stack_push(arguments)?;
    process.stack_push(function)?;
    process.stack_push(module)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

/// `module`, `function`, and arity of `argument_list` must have code registered with
/// `crate::runtime::code::export::insert` or returns `undef` exception.
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let module = arc_process.stack_peek(1).unwrap();
    let function = arc_process.stack_peek(2).unwrap();
    let argument_list = arc_process.stack_peek(3).unwrap();

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

    let module_atom: Atom = module.try_into().unwrap();
    let function_atom: Atom = function.try_into().unwrap();
    let arity: Arity = argument_vec.len().try_into().unwrap();

    match crate::runtime::code::export::get(&module_atom, &function_atom, arity) {
        Some(code) => {
            arc_process.stack_popn(3);

            crate::runtime::code::export::place_frame_with_arguments(
                &arc_process,
                Placement::Replace,
                module_atom,
                function_atom,
                arity,
                code,
                argument_vec,
            )?;

            Process::call_code(arc_process)
        }
        None => undef(
            arc_process,
            module,
            function,
            argument_list,
            anyhow!(
                ":{}.{}/{} is not exported",
                module_atom.name(),
                function_atom.name(),
                arity
            )
            .into(),
            3,
        ),
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

fn undef(
    arc_process: &Arc<Process>,
    module: Term,
    function: Term,
    arguments: Term,
    source: ArcError,
    stack_used: usize,
) -> code::Result {
    arc_process.reduce();
    let exception = exception::undef(arc_process, module, function, arguments, Term::NIL, source);
    code::result_from_exception(arc_process, stack_used, exception)
}

lazy_static! {
    static ref RW_LOCK_CODE: RwLock<Code> = RwLock::new(code);
}

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Term, TypedTerm};
use liblumen_alloc::{Arity, ModuleFunctionArity};

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

/// `module`, `function`, and arity of `argument_list` must have code registered with
/// `crate::code::export::insert` or returns `undef` exception.
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();

    match argument_list.to_typed_term().unwrap() {
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

    match crate::code::export::get(&module_atom, &function_atom, arity) {
        Some(code) => {
            crate::code::export::place_frame_with_arguments(
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
        None => undef(arc_process, module, function, argument_list),
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

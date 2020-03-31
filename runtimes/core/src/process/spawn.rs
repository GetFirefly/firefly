pub mod options;

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{CloneToProcess, ModuleFunctionArity};

use crate::code::export;

pub use self::options::{Connection, Options};

/// Spawns a process with arguments for `apply(module, function, arguments)` on its stack.
///
/// This allows the `apply/3` code to be changed with `apply_3::set_code(code)` to handle new
/// MFA unique to a given application.
pub fn apply_3(
    parent_process: &Process,
    options: Options,
    module: Atom,
    function: Atom,
    arguments: Term,
) -> exception::Result<Spawned> {
    let arity = arity(arguments);

    let child_process = options.spawn(Some(parent_process), module, function, arity)?;

    let module_term = module.encode()?;
    let function_term = function.encode()?;
    let heap_arguments = arguments.clone_to_process(&child_process);

    let erlang_atom = Atom::try_from_str("erlang").unwrap();
    let apply_atom = Atom::try_from_str("apply").unwrap();
    let arity = 3;
    let code = export::get(&erlang_atom, &apply_atom, arity).expect("erlang:apply/3 not exported");
    child_process.stack_push(heap_arguments)?;
    child_process.stack_push(function_term)?;
    child_process.stack_push(module_term)?;
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: erlang_atom,
        function: apply_atom,
        arity,
    });
    let frame = Frame::new(module_function_arity, code);
    child_process.place_frame(frame, Placement::Push);

    // Connect after placing frame, so that any logging can show the `Frame`s when connections occur
    let connection = options.connect(Some(&parent_process), &child_process)?;

    Ok(Spawned {
        process: child_process,
        connection,
    })
}

/// Spawns a process with `arguments` on its stack and `code` run with those arguments instead
/// of passing through `apply/3`.
pub fn code(
    parent_process: Option<&Process>,
    options: Options,
    module: Atom,
    function: Atom,
    arguments: &[Term],
    code: Code,
) -> exception::Result<Spawned> {
    let arity = arguments.len() as u8;

    let child_process = options.spawn(parent_process, module, function, arity)?;

    for argument in arguments.iter().rev() {
        let process_argument = argument.clone_to_process(&child_process);
        child_process.stack_push(process_argument)?;
    }

    let frame = Frame::new(child_process.initial_module_function_arity.clone(), code);
    child_process.push_frame(frame);

    // Connect after placing frame, so that any logging can show the `Frame`s when connections occur
    let connection = options.connect(parent_process, &child_process)?;

    Ok(Spawned {
        process: child_process,
        connection,
    })
}

pub struct Spawned {
    pub process: Process,
    #[must_use]
    pub connection: Connection,
}

// Private

fn arity(arguments: Term) -> u8 {
    match arguments.decode().unwrap() {
        TypedTerm::Nil => 0,
        TypedTerm::List(cons) => cons.count().unwrap().try_into().unwrap(),
        _ => {
            panic!(
                "Arguments {:?} are neither an empty nor a proper list",
                arguments
            );
        }
    }
}

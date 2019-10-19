pub mod options;

use std::convert::TryInto;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::Code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{AsTerm, Atom, Term, TypedTerm};
use liblumen_alloc::CloneToProcess;

use crate::otp::erlang;
use crate::process::spawn::options::{Connection, Options};

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
) -> Result<Spawned, Alloc> {
    let arity = arity(arguments);

    let child_process = options.spawn(Some(parent_process), module, function, arity)?;

    let module_term = unsafe { module.as_term() };
    let function_term = unsafe { function.as_term() };
    let heap_arguments = arguments.clone_to_process(&child_process);

    erlang::apply_3::place_frame_with_arguments(
        &child_process,
        Placement::Push,
        module_term,
        function_term,
        heap_arguments,
    )?;

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
) -> Result<Spawned, Alloc> {
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
    match arguments.to_typed_term().unwrap() {
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

#[cfg(test)]
use core::convert::TryInto;

use alloc::sync::Arc;

use liblumen_core::locks::RwLock;

use liblumen_alloc::erts::exception::{self, Exception};
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::{self, Code};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::Atom;
#[cfg(test)]
use liblumen_alloc::erts::term::{Boxed, Cons, Term};
use liblumen_alloc::erts::ModuleFunctionArity;
use liblumen_alloc::undef;

#[cfg(test)]
use crate::otp::erlang;

/// A stub that just puts the init process into `Status::Waiting`, so it remains alive without
/// wasting CPU cycles
pub fn init(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    Arc::clone(arc_process).wait();

    Ok(())
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
pub fn apply_fn() -> Code {
    *RW_LOCK_APPLY.read()
}

pub fn set_apply_fn(code: Code) {
    *RW_LOCK_APPLY.write() = code;
}

/// Treats all MFAs as undefined.
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
/// #### Process
///
/// * `status` - `Status::Exiting` with `undef!` exception.
#[cfg(not(test))]
fn apply(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    // arguments are consumed, but unused
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let arguments = arc_process.stack_pop().unwrap();
    arc_process.reduce();

    match undef!(&mut arc_process.acquire_heap(), module, function, arguments) {
        Exception::Runtime(runtime_exception) => {
            arc_process.exception(runtime_exception);

            Ok(())
        }
        Exception::System(system_exception) => Err(system_exception),
    }
}

// I have no idea how this would work in LLVM generated code, but for testing `spawn/3` this allows
// `crate::instructions::Instructions::Apply` to translate Terms to Rust functions.
#[cfg(test)]
pub fn apply(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    let module = arc_process.stack_pop().unwrap();
    let function = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();
    let argument_cons: Boxed<Cons> = argument_list.try_into().unwrap();

    for result in argument_cons.into_iter() {
        let element = result.unwrap();

        argument_vec.push(element);
    }

    let arity = argument_vec.len();

    let module_atom: Atom = module.try_into().unwrap();
    let function_atom: Atom = function.try_into().unwrap();

    let result = match module_atom.name() {
        "erlang" => match function_atom.name() {
            "+" => match arity {
                1 => erlang::number_or_badarith_1(argument_vec[0]),
                _ => Err(undef!(
                    &mut arc_process.acquire_heap(),
                    module,
                    function,
                    argument_list
                )),
            },
            "self" => match arity {
                0 => Ok(erlang::self_0(arc_process)),
                _ => Err(undef!(
                    &mut arc_process.acquire_heap(),
                    module,
                    function,
                    argument_list
                )
                .into()),
            },
            _ => Err(undef!(
                &mut arc_process.acquire_heap(),
                module,
                function,
                argument_list
            )
            .into()),
        },
        _ => Err(undef!(
            &mut arc_process.acquire_heap(),
            module,
            function,
            argument_list
        )
        .into()),
    };

    arc_process.reduce();

    match result {
        Ok(term) => {
            // Exception outlives the stack frame, so it can be used to pass data back to the test
            arc_process.exception(liblumen_alloc::exit!(term));

            Ok(())
        }
        Err(exception) => match exception {
            Exception::Runtime(runtime_exception) => {
                arc_process.exception(runtime_exception);

                Ok(())
            }
            Exception::System(system_exception) => Err(system_exception),
        },
    }
}

/// Adds a frame for the BIF so that stacktraces include the BIF
pub fn tail_call_bif<F>(
    arc_process: &Arc<ProcessControlBlock>,
    module: Atom,
    function: Atom,
    arity: u8,
    bif: F,
) -> code::Result
where
    F: Fn() -> exception::Result,
{
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module,
        function,
        arity,
    });
    let frame = Frame::new(
        module_function_arity,
        // fake code
        apply_fn(),
    );

    // fake frame show BIF shows in stacktraces
    arc_process.push_frame(frame);

    match bif() {
        Ok(term) => {
            arc_process.reduce();

            // remove BIF frame before returning from call, so that caller's caller is invoked
            // by `call_code`
            arc_process.pop_code_stack();
            arc_process.return_from_call(term)?;

            ProcessControlBlock::call_code(arc_process)
        }
        Err(exception) => {
            arc_process.reduce();

            match exception {
                Exception::Runtime(runtime_exception) => {
                    arc_process.exception(runtime_exception);

                    Ok(())
                }
                Exception::System(system_exception) => Err(system_exception),
            }
        }
    }
}

lazy_static! {
    static ref RW_LOCK_APPLY: RwLock<Code> = RwLock::new(apply);
}

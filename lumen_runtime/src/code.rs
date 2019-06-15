#[cfg(test)]
use std::convert::TryInto;
use std::sync::{Arc, RwLock};

#[cfg(test)]
use crate::otp::erlang;
use crate::process::Process;
#[cfg(test)]
use crate::term::Term;

pub type Code = fn(&Arc<Process>) -> ();

/// A stub that just puts the init process into `Status::Waiting`, so it remains alive without
/// wasting CPU cycles
pub fn init(arc_process: &Arc<Process>) {
    Arc::clone(arc_process).wait();
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
    *RW_LOCK_APPLY.read().unwrap()
}

pub fn set_apply_fn(code: Code) {
    *RW_LOCK_APPLY.write().unwrap() = code;
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
fn apply(arc_process: &Arc<Process>) {
    // arguments are consumed, but unused
    let argument_vec = arc_process.pop_arguments(3);
    let module = argument_vec[0];
    let function = argument_vec[1];
    let arguments = argument_vec[2];
    arc_process.reduce();

    arc_process.exception(undef!(module, function, arguments, arc_process));
}

// I have no idea how this would work in LLVM generated code, but for testing `spawn/3` this allows
// `crate::instructions::Instructions::Apply` to translate Terms to Rust functions.
#[cfg(test)]
pub fn apply(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(3);
    let module = frame_argument_vec[0];
    let function = frame_argument_vec[1];
    let argument_list = frame_argument_vec[2];

    let argument_vec: Vec<Term> = argument_list.try_into().unwrap();
    let arity = argument_vec.len();

    let result = match unsafe { module.atom_to_string() }.as_ref().as_ref() {
        "erlang" => match unsafe { function.atom_to_string() }.as_ref().as_ref() {
            "+" => match arity {
                1 => erlang::number_or_badarith_1(argument_vec[0]),
                _ => Err(undef!(module, function, argument_list, arc_process)),
            },
            "self" => match arity {
                0 => Ok(erlang::self_0(arc_process)),
                _ => Err(undef!(module, function, argument_list, arc_process)),
            },
            _ => Err(undef!(module, function, argument_list, arc_process)),
        },
        _ => Err(undef!(module, function, argument_list, arc_process)),
    };

    arc_process.reduce();

    let exception = match result {
        Ok(term) => {
            // Exception outlives the stack frame, so it can be used to pass data back to the test
            exit!(term)
        }
        Err(err_exception) => err_exception,
    };

    arc_process.exception(exception);
}

lazy_static! {
    static ref RW_LOCK_APPLY: RwLock<Code> = RwLock::new(apply);
}

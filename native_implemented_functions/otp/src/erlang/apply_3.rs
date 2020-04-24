use std::convert::TryInto;
use std::ffi::c_void;
use std::mem::transmute;

use anyhow::*;
use lazy_static::lazy_static;

use liblumen_core::locks::RwLock;
use liblumen_core::symbols::FunctionSymbol;
use liblumen_core::sys::dynamic_call::DynamicCallee;

use liblumen_alloc::erts::apply::find_symbol;
use liblumen_alloc::erts::exception::{self, ArcError};
use liblumen_alloc::erts::process::{Frame, FrameWithArguments, Native};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::{Arity, ModuleFunctionArity};

use crate::runtime::process::current_process;

const ARITY: Arity = 3;

pub fn function_symbol() -> FunctionSymbol {
    FunctionSymbol {
        module: super::module_id(),
        function: function().id(),
        arity: ARITY,
        ptr: get_native() as *const c_void,
    }
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
/// * `status` - `Status::RuntimeException` with exception from lookup or called function.
pub fn get_native() -> extern "C" fn(Term, Term, Term) -> Term {
    *RW_LOCK_NATIVE.read()
}

pub fn set_native(native: extern "C" fn(Term, Term, Term) -> Term) {
    *RW_LOCK_NATIVE.write() = native;
}

pub fn frame_with_arguments(module: Term, function: Term, arguments: Term) -> FrameWithArguments {
    frame().with_arguments(false, &[module, function, arguments])
}

// Private

/// `module`, `function`, and arity of `argument_list` must have code registered with
/// `crate::runtime::code::export::insert` or returns `undef` exception.
pub extern "C" fn native(module: Term, function: Term, argument_list: Term) -> Term {
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
    let module_function_arity = ModuleFunctionArity {
        module: module_atom,
        function: function_atom,
        arity,
    };

    match find_symbol(&module_function_arity) {
        Some(dynamic_call) => {
            let native = unsafe {
                let ptr = transmute::<DynamicCallee, *const c_void>(dynamic_call);

                Native::from_ptr(ptr, arity)
            };
            let frame = Frame::new(module_function_arity, native);
            let frame_with_arguments = frame.with_arguments(false, &argument_vec);

            current_process().queue_frame_with_arguments(frame_with_arguments);

            Term::NONE
        }
        None => undef(
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
        ),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), Native::Three(get_native()))
}

fn function() -> Atom {
    Atom::try_from_str("apply").unwrap()
}

pub(crate) fn module_function_arity() -> ModuleFunctionArity {
    ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 3,
    }
}

fn undef(module: Term, function: Term, arguments: Term, source: ArcError) -> Term {
    let arc_process = current_process();
    arc_process.reduce();

    let exception_result = Err(exception::undef(
        &arc_process,
        module,
        function,
        arguments,
        Term::NIL,
        source,
    ));

    arc_process.return_status(exception_result)
}

lazy_static! {
    static ref RW_LOCK_NATIVE: RwLock<extern "C" fn(Term, Term, Term) -> Term> =
        RwLock::new(native);
}

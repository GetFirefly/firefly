use std::convert::TryInto;
use std::sync::Arc;

use cranelift_entity::EntityRef;
use libeir_ir::Block;

use liblumen_alloc::erts::exception::runtime;
use liblumen_alloc::erts::process::code::result_from_exception;
use liblumen_alloc::erts::process::code::stack::frame::Frame;
use liblumen_alloc::erts::process::code::Result;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::exec::CallExecutor;

pub fn return_clean(arc_process: &Arc<Process>) -> Result {
    let argument_list = arc_process.stack_pop().unwrap();
    arc_process.return_from_call(argument_list)?;
    Process::call_code(arc_process)
}

pub fn return_ok(arc_process: &Arc<Process>) -> Result {
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
        _ => panic!(),
    }
    assert!(argument_vec.len() == 1);

    Ok(arc_process.return_from_call(argument_vec[0])?)
}

pub fn return_throw(arc_process: &Arc<Process>) -> Result {
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
        _ => panic!(),
    }

    let class: Atom = argument_vec[0].try_into().unwrap();
    let class = match class.name() {
        "EXIT" => runtime::Class::Exit,
        "throw" => runtime::Class::Throw,
        "error" => runtime::Class::Error { arguments: None },
        k => unreachable!("{:?}", k),
    };

    let exc = runtime::Exception {
        class,
        reason: argument_vec[1],
        stacktrace: Some(argument_vec[2]),
        file: "",
        line: 0,
        column: 0,
    };
    result_from_exception(arc_process, exc.into())
}

/// Expects the following on stack:
/// * arity integer
/// * argument list
pub fn interpreter_mfa_code(arc_process: &Arc<Process>) -> Result {
    let argument_list = arc_process.stack_pop().unwrap();

    let mfa = arc_process.current_module_function_arity().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();
    match argument_list.to_typed_term().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!(),
    }
    assert!(mfa.arity as usize == argument_vec.len() - 2);

    let mut exec = CallExecutor::new();
    exec.call(
        &crate::VM,
        arc_process,
        mfa.module,
        mfa.function,
        argument_vec.len() - 2,
        &mut argument_vec,
    );

    Ok(())
}

/// Expects the following on stack:
/// * arity integer
/// * argument list
/// * block id integer
/// * environment list
pub fn interpreter_closure_code(arc_process: &Arc<Process>) -> Result {
    let argument_list = arc_process.stack_pop().unwrap();
    let closure_term = arc_process.stack_pop().unwrap();

    let closure: Boxed<Closure> = closure_term.try_into().unwrap();

    let mfa = arc_process.current_module_function_arity().unwrap();
    let arity = mfa.arity;

    let block_id: usize = closure.env_slice()[0].try_into().unwrap();
    let block = Block::new(block_id);

    let mut argument_vec: Vec<Term> = Vec::new();
    match argument_list.to_typed_term().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!(),
    }

    let mut environment_vec: Vec<Term> = closure.env_slice()[1..].to_owned();

    let mut exec = CallExecutor::new();
    exec.call_block(
        &crate::VM,
        arc_process,
        mfa.module,
        mfa.function,
        arity as usize,
        &mut argument_vec,
        block,
        &mut environment_vec,
    );

    Ok(())
}

pub fn apply(arc_process: &Arc<Process>) -> Result {
    let module_term = arc_process.stack_pop().unwrap();
    let function_term = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let module: Atom = module_term.try_into().unwrap();
    let function: Atom = function_term.try_into().unwrap();

    let arity;
    match argument_list.to_typed_term().unwrap() {
        TypedTerm::Nil => panic!(),
        TypedTerm::List(argument_cons) => arity = argument_cons.into_iter().count() - 2,
        _ => panic!(),
    }

    let module_function_arity = Arc::new(ModuleFunctionArity {
        module,
        function,
        arity: arity.try_into().unwrap(),
    });

    let frame = Frame::new(module_function_arity, interpreter_mfa_code);

    arc_process.stack_push(argument_list)?;
    arc_process.replace_frame(frame);

    Process::call_code(arc_process)
}

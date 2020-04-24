use std::convert::TryInto;
use std::sync::Arc;

use anyhow::*;
use cranelift_entity::EntityRef;
use libeir_ir::{Block, FunctionIndex};

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::frames;
use liblumen_alloc::erts::process::frames::stack::frame::Frame;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::exec::CallExecutor;

fn module() -> Atom {
    Atom::try_from_str("lumen_eir_interpreter_intrinsics").unwrap()
}

pub fn return_clean(arc_process: &Arc<Process>) -> frames::Result {
    let argument_list = arc_process.stack_pop().unwrap();
    arc_process.return_from_call(0, argument_list)?;
    Process::call_native_or_yield(arc_process)
}

pub fn return_clean_closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_clean").unwrap();
    const ARITY: u8 = 1;

    process
        .export_closure(module(), function, ARITY, Some(return_clean))
        .map_err(|error| error.into())
}

pub fn return_ok(arc_process: &Arc<Process>) -> frames::Result {
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
        _ => panic!(),
    }
    assert!(argument_vec.len() == 1);

    Ok(arc_process.return_from_call(0, argument_vec[0])?)
}

pub fn return_ok_closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_ok").unwrap();
    const ARITY: u8 = 1;

    process
        .export_closure(module(), function, ARITY, Some(return_ok))
        .map_err(|error| error.into())
}

pub fn return_throw(arc_process: &Arc<Process>) -> frames::Result {
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
        _ => panic!(),
    }

    let class: exception::Class = argument_vec[0].try_into().unwrap();

    let reason = argument_vec[1];
    let stacktrace = Some(argument_vec[2]);
    let exception = exception::raise(
        class,
        reason,
        stacktrace,
        anyhow!("explicit raise from Erlang").into(),
    );
    frames::exception_to_native_return(arc_process, 0, exception.into())
}

pub fn return_throw_closure(process: &Process) -> exception::Result<Term> {
    let function = Atom::try_from_str("return_throw").unwrap();
    const ARITY: u8 = 3;

    process
        .export_closure(module(), function, ARITY, Some(return_throw))
        .map_err(|error| error.into())
}

/// Expects the following on stack:
/// * arity integer
/// * argument list
pub fn interpreter_mfa_code(arc_process: &Arc<Process>) -> frames::Result {
    let argument_list = arc_process.stack_pop().unwrap();

    let mfa = arc_process.current_module_function_arity().unwrap();

    let mut argument_vec: Vec<Term> = Vec::new();
    match argument_list.decode().unwrap() {
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
pub fn interpreter_closure_code(arc_process: &Arc<Process>) -> frames::Result {
    let argument_list = arc_process.stack_pop().unwrap();
    let closure_term = arc_process.stack_pop().unwrap();

    let closure: Boxed<Closure> = closure_term.try_into().unwrap();
    println!("{:?}", closure);

    let mfa = arc_process.current_module_function_arity().unwrap();
    let definition = arc_process.current_definition().unwrap();

    let block_id;
    let function_index;
    match definition {
        Definition::Anonymous {
            index, old_unique, ..
        } => {
            block_id = index;
            function_index = old_unique;
        }
        _ => unreachable!(),
    }

    //let block_id: usize = closure.env_slice()[0].try_into().unwrap();
    let block = Block::new(block_id as usize);

    let mut argument_vec: Vec<Term> = Vec::new();
    match argument_list.decode().unwrap() {
        TypedTerm::Nil => (),
        TypedTerm::List(argument_cons) => {
            for result in argument_cons.into_iter() {
                let element = result.unwrap();

                argument_vec.push(element);
            }
        }
        _ => panic!(),
    }

    let mut environment_vec: Vec<Term> = closure.env_slice().to_owned();

    let mut exec = CallExecutor::new();
    exec.call_block(
        &crate::VM,
        arc_process,
        mfa.module,
        FunctionIndex::new(function_index as usize),
        //mfa.function,
        //arity as usize,
        &mut argument_vec,
        block,
        &mut environment_vec,
    );

    Ok(())
}

pub fn apply(arc_process: &Arc<Process>) -> frames::Result {
    let module_term = arc_process.stack_pop().unwrap();
    let function_term = arc_process.stack_pop().unwrap();
    let argument_list = arc_process.stack_pop().unwrap();

    let module: Atom = module_term.try_into().unwrap();
    let function: Atom = function_term.try_into().unwrap();

    let arity;
    match argument_list.decode().unwrap() {
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

    Process::call_native_or_yield(arc_process)
}

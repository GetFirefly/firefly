use std::convert::TryInto;
use std::sync::Arc;

use cranelift_entity::EntityRef;

use libeir_ir::{Block, FunctionIndex};

use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;

use locate_code::locate_code;

use crate::exec::CallExecutor;

/// Expects the following on stack:
/// * arity integer
/// * argument list
/// * block id integer
/// * environment list
#[locate_code]
pub fn code(arc_process: &Arc<Process>) -> code::Result {
    let argument_list = arc_process.stack_pop().unwrap();
    let closure_term = arc_process.stack_pop().unwrap();

    let closure: Boxed<Closure> = closure_term.try_into().unwrap();
    println!("{:?}", closure);

    let frame = arc_process.current_frame().unwrap();
    let definition = frame.definition;

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
        frame.module,
        FunctionIndex::new(function_index as usize),
        //mfa.function,
        //arity as usize,
        &mut argument_vec,
        block,
        &mut environment_vec,
    );

    Ok(())
}

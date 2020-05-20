use std::convert::TryInto;
use std::sync::Arc;

use cranelift_entity::EntityRef;

use libeir_ir::{Block, FunctionIndex};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::closure::Definition;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::exec::CallExecutor;

/// Expects the following on stack:
/// * arity integer
/// * argument list
/// * block id integer
/// * environment list
#[native_implemented_function(interpreter_closure/2)]
pub fn result(arc_process: Arc<Process>, argument_list: Term, closure_term: Term) -> Term {
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
        &arc_process,
        mfa.module,
        FunctionIndex::new(function_index as usize),
        //mfa.function,
        //arity as usize,
        &mut argument_vec,
        block,
        &mut environment_vec,
    );

    Term::NONE
}

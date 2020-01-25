use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::borrow::clone_to_process::CloneToProcess;
use liblumen_alloc::erts::process::code;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::HeapFragment;

use locate_code::locate_code;

use super::{ProcessResult, ProcessResultSender};

#[locate_code]
fn code(arc_process: &Arc<Process>) -> code::Result {
    let argument_list = arc_process.stack_pop().unwrap();
    let closure_term = arc_process.stack_pop().unwrap();

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

    let closure: Boxed<Closure> = closure_term.try_into().unwrap();
    let sender_resource: Boxed<Resource> = closure.env_slice()[0].try_into().unwrap();
    let sender_any: Resource = sender_resource.into();
    let sender: &ProcessResultSender = sender_any.downcast_ref().unwrap();

    let mut fragment = HeapFragment::new_from_word_size(100).unwrap();
    let frag_mut = unsafe { fragment.as_mut() };
    let ret = argument_vec[0].clone_to_heap(frag_mut).unwrap();

    sender
        .tx
        .send(ProcessResult {
            heap: fragment,
            result: Ok(ret),
        })
        .unwrap();

    Ok(arc_process.return_from_call(0, argument_vec[0])?)
}

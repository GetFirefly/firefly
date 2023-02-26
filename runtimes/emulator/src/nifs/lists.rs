use firefly_alloc::heap::Heap;
use firefly_rt::function::ErlangResult;
use firefly_rt::gc::{garbage_collect, Gc, RootSet};
use firefly_rt::process::ProcessLock;
use firefly_rt::term::*;

use crate::badarg;

#[export_name = "lists:reverse/2"]
pub extern "C-unwind" fn reverse(
    process: &mut ProcessLock,
    mut list: OpaqueTerm,
    mut tail: OpaqueTerm,
) -> ErlangResult {
    // If we get a non-empty list, we can return the tail directly
    if list.is_nil() {
        return ErlangResult::Ok(tail);
    }

    let heap_top = process.heap_top();
    match list.into() {
        Term::Cons(cons) => {
            'retry: loop {
                let mut builder = if tail.is_nil() {
                    ListBuilder::new(process)
                } else if tail.is_nonempty_list() {
                    let tail = unsafe { Gc::from_raw(tail.as_ptr() as *mut Cons) };
                    ListBuilder::prepend(tail, process)
                } else {
                    ListBuilder::new_improper(tail, process)
                };
                for item in cons.iter() {
                    match item {
                        Ok(term) => {
                            if let Err(_) = unsafe { builder.push_unsafe(term) } {
                                let mut roots = RootSet::default();
                                roots += &mut list as *mut _;
                                roots += &mut tail as *mut _;
                                assert!(garbage_collect(process, roots).is_ok());
                                continue 'retry;
                            }
                        }
                        Err(_improper) => {
                            // Reset the heap as we aren't going to use the cells we've allocated
                            unsafe {
                                process.reset_heap_top(heap_top);
                            }
                            badarg!(process, list);
                        }
                    }
                }
                // We are guaranteed to have a cons cell here
                return ErlangResult::Ok(builder.finish().unwrap().into());
            }
        }
        _other => badarg!(process, list),
    }
}

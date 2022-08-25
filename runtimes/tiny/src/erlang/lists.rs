use std::ops::Deref;

use liblumen_rt::backtrace::Trace;
use liblumen_rt::function::ErlangResult;
use liblumen_rt::term::*;

use crate::scheduler;

use super::badarg;

#[export_name = "lists:reverse/2"]
#[allow(improper_ctypes_definitions)]
pub extern "C-unwind" fn reverse(list: OpaqueTerm, tail: OpaqueTerm) -> ErlangResult {
    // If we get a non-empty list, we can return the tail directly
    if list.is_nil() {
        return ErlangResult::Ok(tail);
    }

    match list.into() {
        Term::Cons(cons) => {
            let cons = unsafe { cons.as_ref() };
            scheduler::with_current(|scheduler| {
                let arc_proc = scheduler.current_process();
                let proc = arc_proc.deref();
                let mut current = None;
                for item in cons.iter() {
                    let head = item
                        .map_err(|_| unsafe { badarg(Trace::capture()).unwrap_err_unchecked() })?;
                    match current.take() {
                        None => {
                            // This is the first iteration
                            let mut ptr = Cons::new_in(proc).unwrap();
                            let cell = unsafe { ptr.as_mut() };
                            cell.tail = tail;
                            cell.head = head.into();
                            current = Some(OpaqueTerm::from(ptr));
                        }
                        Some(tail) => {
                            let mut next_ptr = Cons::new_in(proc).unwrap();
                            let cell = unsafe { next_ptr.as_mut() };
                            cell.tail = tail;
                            cell.head = head.into();
                            current = Some(OpaqueTerm::from(next_ptr));
                        }
                    }
                }
                // We know we have at least one cell because the list in this branch is nonempty
                ErlangResult::Ok(current.unwrap())
            })
        }
        _other => badarg(Trace::capture()),
    }
}

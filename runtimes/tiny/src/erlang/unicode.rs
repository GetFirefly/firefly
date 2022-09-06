use std::collections::VecDeque;
use std::ops::Deref;

use firefly_binary::{BitVec, Bitstring, Encoding};
use firefly_rt::backtrace::Trace;
use firefly_rt::function::ErlangResult;
use firefly_rt::term::*;

use crate::scheduler;

use super::badarg;

#[export_name = "unicode:characters_to_list/2"]
pub extern "C-unwind" fn characters_to_list(
    data: OpaqueTerm,
    encoding: OpaqueTerm,
) -> ErlangResult {
    let Term::Atom(encoding) = encoding.into() else { return badarg(Trace::capture()) };
    let Ok(encoding) = encoding.as_str().parse::<Encoding>() else { return badarg(Trace::capture()) };

    // Convert the input iodata into a binary
    let mut buffer = BitVec::new();

    let latin1 = encoding == Encoding::Latin1 || encoding == Encoding::Raw;

    let mut queue = VecDeque::new();
    queue.push_back(data.into());
    while let Some(term) = queue.pop_front() {
        match term {
            Term::Nil => continue,
            Term::Cons(ptr) => {
                let cons = unsafe { ptr.as_ref() };
                let mut items = flat_map_maybe_improper(cons, vec![]);
                queue.extend(items.drain(..));
            }
            Term::Int(codepoint) if latin1 => {
                // TODO: validate codepoint
                let byte: u8 = codepoint.try_into().unwrap();
                buffer.push_byte(byte);
            }
            Term::Int(codepoint) if codepoint >= 0 && codepoint < u32::MAX as i64 => {
                // TODO: validate codepoint
                let c: char = (codepoint as u32).try_into().unwrap();
                buffer.push_utf8(c);
            }
            // TODO: Raise proper error
            _invalid => return badarg(Trace::capture()),
        }
    }

    // Then convert the binary to a list of codepoints
    match buffer.as_str() {
        Some(s) => scheduler::with_current(|scheduler| {
            let arc_proc = scheduler.current_process();
            let proc = arc_proc.deref();
            ErlangResult::Ok(
                Cons::charlist_from_str(s, proc)
                    .unwrap()
                    .map(Term::Cons)
                    .unwrap_or(Term::Nil)
                    .into(),
            )
        }),
        None => {
            // There must be valid latin1, but invalid unicode codepoints
            assert!(latin1, "invalid unicode codepoints in binary");
            scheduler::with_current(|scheduler| {
                let arc_proc = scheduler.current_process();
                let proc = arc_proc.deref();
                ErlangResult::Ok(
                    Cons::from_bytes(unsafe { buffer.as_bytes_unchecked() }, proc)
                        .unwrap()
                        .map(Term::Cons)
                        .unwrap_or(Term::Nil)
                        .into(),
                )
            })
        }
    }
}

fn flat_map_maybe_improper(cons: &Cons, acc: Vec<Term>) -> Vec<Term> {
    cons.iter()
        .fold(acc, |mut acc, maybe_improper| match maybe_improper {
            Ok(Term::Cons(ptr)) => {
                let nested = unsafe { ptr.as_ref() };
                flat_map_maybe_improper(nested, acc)
            }
            Ok(term) => {
                acc.push(term);
                acc
            }
            Err(improper) => {
                acc.push(improper.tail);
                acc
            }
        })
}

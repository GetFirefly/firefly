use std::collections::VecDeque;

use firefly_binary::{BitVec, Bitstring, Encoding};
use firefly_rt::function::ErlangResult;
use firefly_rt::gc::garbage_collect;
use firefly_rt::process::ProcessLock;
use firefly_rt::term::*;

use crate::badarg;

#[export_name = "unicode:characters_to_list/2"]
pub extern "C-unwind" fn characters_to_list(
    process: &mut ProcessLock,
    data: OpaqueTerm,
    encoding: OpaqueTerm,
) -> ErlangResult {
    let Term::Atom(e) = encoding.into() else { badarg!(process, encoding) };
    let Ok(encoding) = e.as_str().parse::<Encoding>() else { badarg!(process, encoding) };

    // Convert the input iodata into a binary
    let mut buffer = BitVec::new();

    let latin1 = encoding == Encoding::Latin1 || encoding == Encoding::Raw;

    let mut queue = VecDeque::new();
    queue.push_back(data.into());
    while let Some(term) = queue.pop_front() {
        match term {
            Term::Nil => continue,
            Term::Cons(cons) => {
                let mut items = flat_map_maybe_improper(&cons, vec![]);
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
            _invalid => badarg!(process, data),
        }
    }

    // Then convert the binary to a list of codepoints
    match buffer.as_str() {
        Some(s) => loop {
            match Cons::charlist_from_str(s, process) {
                Ok(charlist) => {
                    return ErlangResult::Ok(charlist.map(Term::Cons).unwrap_or(Term::Nil).into());
                }
                Err(_) => {
                    // We don't need to pass any roots, because all our live data is in the BitVec
                    assert!(garbage_collect(process, Default::default()).is_ok());
                }
            }
        },
        None => loop {
            // There must be valid latin1, but invalid unicode codepoints
            assert!(latin1, "invalid unicode codepoints in binary");
            match Cons::from_bytes(unsafe { buffer.as_bytes_unchecked() }, process) {
                Ok(charlist) => {
                    return ErlangResult::Ok(charlist.map(Term::Cons).unwrap_or(Term::Nil).into());
                }
                Err(_) => {
                    // We don't need to pass any roots, because all our live data is in the BitVec
                    assert!(garbage_collect(process, Default::default()).is_ok());
                }
            }
        },
    }
}

fn flat_map_maybe_improper(cons: &Cons, acc: Vec<Term>) -> Vec<Term> {
    cons.iter()
        .fold(acc, |mut acc, maybe_improper| match maybe_improper {
            Ok(Term::Cons(nested)) => flat_map_maybe_improper(&nested, acc),
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

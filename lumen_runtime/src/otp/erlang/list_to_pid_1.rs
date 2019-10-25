// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(list_to_pid/1)]
pub fn native(process: &Process, string: Term) -> exception::Result {
    let cons: Boxed<Cons> = string.try_into()?;

    let prefix_tail = skip_char(cons, '<')?;
    let prefix_tail_cons: Boxed<Cons> = prefix_tail.try_into()?;

    let (node_id, node_tail) = next_decimal(prefix_tail_cons)?;
    let node_tail_cons: Boxed<Cons> = node_tail.try_into()?;

    let first_separator_tail = skip_char(node_tail_cons, '.')?;
    let first_separator_tail_cons: Boxed<Cons> = first_separator_tail.try_into()?;

    let (number, number_tail) = next_decimal(first_separator_tail_cons)?;
    let number_tail_cons: Boxed<Cons> = number_tail.try_into()?;

    let second_separator_tail = skip_char(number_tail_cons, '.')?;
    let second_separator_tail_cons: Boxed<Cons> = second_separator_tail.try_into()?;

    let (serial, serial_tail) = next_decimal(second_separator_tail_cons)?;
    let serial_tail_cons: Boxed<Cons> = serial_tail.try_into()?;

    let suffix_tail = skip_char(serial_tail_cons, '>')?;

    if suffix_tail.is_nil() {
        process
            .pid_with_node_id(node_id, number, serial)
            .map_err(|error| error.into())
    } else {
        Err(badarg!().into())
    }
}

// Private

fn next_decimal(cons: Boxed<Cons>) -> Result<(usize, Term), exception::Exception> {
    next_decimal_digit(cons)
        .and_then(|(first_digit, first_tail)| rest_decimal_digits(first_digit, first_tail))
}

fn next_decimal_digit(cons: Boxed<Cons>) -> Result<(u8, Term), exception::Exception> {
    let head_char: char = cons.head.try_into()?;

    match head_char.to_digit(10) {
        Some(digit) => Ok((digit as u8, cons.tail)),
        None => Err(badarg!().into()),
    }
}

fn rest_decimal_digits(
    first_digit: u8,
    first_tail: Term,
) -> Result<(usize, Term), exception::Exception> {
    match first_tail.try_into() {
        Ok(first_tail_cons) => {
            let mut acc_decimal: usize = first_digit as usize;
            let mut acc_tail = first_tail;
            let mut acc_cons: Boxed<Cons> = first_tail_cons;

            while let Ok((digit, tail)) = next_decimal_digit(acc_cons) {
                acc_decimal = 10 * acc_decimal + (digit as usize);
                acc_tail = tail;

                match tail.try_into() {
                    Ok(tail_cons) => acc_cons = tail_cons,
                    Err(_) => {
                        break;
                    }
                }
            }

            Ok((acc_decimal, acc_tail))
        }
        Err(_) => Ok((first_digit as usize, first_tail)),
    }
}

fn skip_char(cons: Boxed<Cons>, skip: char) -> exception::Result {
    let c: char = cons.head.try_into()?;

    if c == skip {
        Ok(cons.tail)
    } else {
        Err(badarg!().into())
    }
}

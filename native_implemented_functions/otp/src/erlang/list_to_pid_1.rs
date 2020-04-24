// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, *};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::distribution::nodes::node;

use crate::runtime::distribution::nodes;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(list_to_pid/1)]
pub fn result(process: &Process, string: Term) -> exception::Result<Term> {
    let cons = term_try_into_non_empty_list!(string)?;

    let prefix_tail = skip_char(cons, '<').context("first character must be '<'")?;
    let prefix_tail_cons =
        try_into_non_empty_list_or_missing(string, prefix_tail, "node.number.serial>")?;

    let (node_id, node_tail) =
        next_decimal(prefix_tail_cons).context("node id must be a decimal integer")?;
    let node_tail_cons = try_into_non_empty_list_or_missing(string, node_tail, ".number.serial>")?;

    let first_separator_tail =
        skip_char(node_tail_cons, '.').context("a '.' must separate the node id and number")?;
    let first_separator_tail_cons =
        try_into_non_empty_list_or_missing(string, first_separator_tail, "number.serial>")?;

    let (number, number_tail) =
        next_decimal(first_separator_tail_cons).context("number must be a decimal integer")?;
    let number_tail_cons = try_into_non_empty_list_or_missing(string, number_tail, ".serial>")?;

    let second_separator_tail =
        skip_char(number_tail_cons, '.').context("a '.' must separate the number and serial")?;
    let second_separator_tail_cons =
        try_into_non_empty_list_or_missing(string, second_separator_tail, "serial>")?;

    let (serial, serial_tail) =
        next_decimal(second_separator_tail_cons).context("serial must be a decimal integer")?;
    let serial_tail_cons = try_into_non_empty_list_or_missing(string, serial_tail, ">")?;

    let suffix_tail = skip_char(serial_tail_cons, '>').context("last character must be '>'")?;

    if suffix_tail.is_nil() {
        if node_id == node::id() {
            Pid::make_term(number, serial).with_context(|| format!("string ({})", string))
        } else {
            let arc_node = nodes::try_id_to_arc_node(&node_id)?;

            process
                .external_pid(arc_node, number, serial)
                .with_context(|| format!("string ({})", string))
        }
    } else {
        Err(TypeError).with_context(|| {
            format!(
                "extra characters ({}) beyond end of formatted pid",
                suffix_tail
            )
        })
    }
    .map_err(From::from)
}

// Private

fn try_into_non_empty_list_or_missing(
    string: Term,
    tail: Term,
    format: &str,
) -> anyhow::Result<Boxed<Cons>> {
    tail.try_into().with_context(|| missing(string, format))
}

fn missing(string: Term, format: &str) -> String {
    format!("string ({}) is missing '{}'", string, format)
}

fn next_decimal(cons: Boxed<Cons>) -> InternalResult<(usize, Term)> {
    next_decimal_digit(cons)
        .and_then(|(first_digit, first_tail)| rest_decimal_digits(first_digit, first_tail))
}

fn next_decimal_digit(cons: Boxed<Cons>) -> InternalResult<(u8, Term)> {
    let head = cons.head;
    let head_char: char = head
        .try_into()
        .context("list element is not a decimal digit")?;

    match head_char.to_digit(10) {
        Some(digit) => Ok((digit as u8, cons.tail)),
        None => Err(anyhow!("{} is not a decimal digit", head).into()),
    }
}

fn rest_decimal_digits(first_digit: u8, first_tail: Term) -> InternalResult<(usize, Term)> {
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

fn skip_char(cons: Boxed<Cons>, skip: char) -> InternalResult<Term> {
    let c: char = cons
        .head
        .try_into()
        .with_context(|| skipped_character(skip))?;

    if c == skip {
        Ok(cons.tail)
    } else {
        Err(TryIntoIntegerError::OutOfRange)
            .with_context(|| skipped_character(skip))
            .map_err(From::from)
    }
}

fn skipped_character(skip: char) -> String {
    format!("skipped character must be {}", skip)
}

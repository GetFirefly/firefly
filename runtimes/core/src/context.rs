pub mod r#type;

use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badmap};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::time;

pub fn string(name: &'static str, quote: char, value: &str) -> String {
    format!("{} ({}{}{})", name, quote, value.escape_default(), quote)
}

pub fn term_is_not_type(name: &str, value: Term, r#type: &str) -> String {
    format!("{} ({}) is not {}", name, value, r#type)
}

pub fn term_is_not_arity(value: Term) -> String {
    term_is_not_type("arity", value, "an arity (an integer in 0-255)")
}

pub fn term_is_not_atom(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "an atom")
}

pub fn term_is_not_boolean(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a boolean")
}

pub fn term_is_not_binary(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a binary")
}

pub fn term_is_not_integer(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "an integer")
}

pub fn term_is_not_number(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a number (integer or float)")
}

pub fn term_is_not_map(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a map")
}

pub fn term_is_not_non_empty_list(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a non-empty list")
}

pub fn term_is_not_non_negative_integer(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a non-negative integer")
}

pub fn term_is_not_one_based_index(index: Term) -> String {
    format!("index ({}) is not a 1-based integer", index)
}

pub fn term_is_not_in_one_based_range(index: Term, max: usize) -> String {
    term_is_not_type(
        "index",
        index,
        &format!("a 1-based integer between 1-{}", max),
    )
}

pub fn term_is_not_pid(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a pid")
}

pub fn term_is_not_reference(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a reference")
}

pub fn term_is_not_time_unit(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a time unit")
}

pub fn term_is_not_tuple(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a tuple")
}

pub fn term_try_into_arity(value: Term) -> anyhow::Result<u8> {
    value.try_into().with_context(|| term_is_not_arity(value))
}

pub fn term_try_into_atom(name: &str, value: Term) -> anyhow::Result<Atom> {
    value
        .try_into()
        .with_context(|| term_is_not_atom(name, value))
}

pub fn term_try_into_bool(name: &str, value: Term) -> anyhow::Result<bool> {
    value
        .try_into()
        .with_context(|| term_is_not_boolean(name, value))
}

pub fn terms_try_into_bools(
    left_name: &str,
    left_value: Term,
    right_name: &str,
    right_value: Term,
) -> anyhow::Result<(bool, bool)> {
    let left_bool = term_try_into_bool(left_name, left_value)?;
    let right_bool = term_try_into_bool(right_name, right_value)?;

    Ok((left_bool, right_bool))
}

pub fn term_try_into_isize(name: &str, value: Term) -> anyhow::Result<isize> {
    value
        .try_into()
        .with_context(|| term_is_not_integer(name, value))
}

pub fn term_try_into_local_pid(name: &str, value: Term) -> anyhow::Result<Pid> {
    value
        .try_into()
        .with_context(|| term_is_not_pid(name, value))
}

pub fn term_try_into_local_reference(name: &str, value: Term) -> anyhow::Result<Boxed<Reference>> {
    value
        .try_into()
        .with_context(|| term_is_not_reference(name, value))
}

pub fn term_try_into_map(name: &str, value: Term) -> anyhow::Result<Boxed<Map>> {
    value
        .try_into()
        .with_context(|| term_is_not_map(name, value))
}

pub fn term_try_into_map_or_badmap(
    process: &Process,
    name: &str,
    value: Term,
) -> exception::Result<Boxed<Map>> {
    term_try_into_map(name, value).map_err(|source| badmap(process, value, source.into()))
}

pub fn term_try_into_non_empty_list(name: &str, value: Term) -> anyhow::Result<Boxed<Cons>> {
    value
        .try_into()
        .with_context(|| term_is_not_non_empty_list(name, value))
}

pub fn term_try_into_one_based_index(index: Term) -> anyhow::Result<OneBasedIndex> {
    index
        .try_into()
        .with_context(|| term_is_not_one_based_index(index))
}

pub fn term_try_into_time_unit(name: &str, value: Term) -> anyhow::Result<time::Unit> {
    value
        .try_into()
        .with_context(|| term_is_not_time_unit(name, value))
}

pub fn term_try_into_tuple(name: &str, value: Term) -> anyhow::Result<Boxed<Tuple>> {
    value
        .try_into()
        .with_context(|| term_is_not_tuple(name, value))
}

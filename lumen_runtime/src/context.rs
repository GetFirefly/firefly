use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::erts::exception::{self, badmap};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn term_is_not_type(name: &str, value: Term, r#type: &str) -> String {
    format!("{} ({}) is not {}", name, value, r#type)
}

pub fn term_is_not_atom(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "an atom")
}

pub fn term_is_not_boolean(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a boolean")
}

pub fn term_is_not_integer(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "an integer")
}

pub fn term_is_not_map(name: &str, value: Term) -> String {
    term_is_not_type(name, value, "a map")
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

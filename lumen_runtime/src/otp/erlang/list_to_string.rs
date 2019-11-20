use std::convert::TryInto;

use anyhow::*;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

pub fn list_to_string(process: &Process, list: Term) -> exception::Result<String> {
    match list.decode().unwrap() {
        TypedTerm::Nil => Ok("".to_owned()),
        TypedTerm::List(cons) => cons
            .into_iter()
            .map(|result| match result {
                Ok(term) => {
                    let c: char = term
                        .try_into()
                        .context("string list elements must be a unicode scalar value")?;

                    Ok(c)
                }
                Err(_) => Err(badarg!(process).into()),
            })
            .collect::<exception::Result<String>>(),
        _ => Err(badarg!(process).into()),
    }
}

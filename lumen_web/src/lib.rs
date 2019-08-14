#![deny(warnings)]

pub mod document;
pub mod element;
pub mod node;
pub mod wait;
pub mod window;

use std::any::Any;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

fn error() -> Term {
    atom_unchecked("error")
}

fn ok() -> Term {
    atom_unchecked("ok")
}

fn ok_tuple(process: &ProcessControlBlock, value: Box<dyn Any>) -> Result<Term, Alloc> {
    let ok = ok();
    let resource_term = process.resource(value)?;

    process.tuple_from_slice(&[ok, resource_term])
}

fn option_to_ok_tuple_or_error<T: 'static>(
    process: &ProcessControlBlock,
    option: Option<T>,
) -> Result<Term, Alloc> {
    match option {
        Some(value) => ok_tuple(process, Box::new(value)),
        None => Ok(error()),
    }
}

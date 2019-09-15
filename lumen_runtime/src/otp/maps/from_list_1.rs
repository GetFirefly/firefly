// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{Atom, Map, Term};
use liblumen_alloc::{badarg, ModuleFunctionArity};

bif!(super::module(), "from_list", native, list);

fn native(process: &Process, list: Term) -> exception::Result {
    match Map::from_list(list) {
        Some(hash_map) => Ok(process.map_from_hash_map(hash_map)?),
        None => Err(badarg!().into()),
    }
}

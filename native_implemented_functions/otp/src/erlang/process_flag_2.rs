// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use anyhow::*;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use native_implemented_function::native_implemented_function;

use crate::runtime::context::*;

#[native_implemented_function(process_flag/2)]
pub fn result(process: &Process, flag: Term, value: Term) -> exception::Result<Term> {
    let flag_atom = term_try_into_atom!(flag)?;

    match flag_atom.name() {
        "error_handler" => unimplemented!(),
        "max_heap_size" => unimplemented!(),
        "message_queue_data" => unimplemented!(),
        "min_bin_vheap_size" => unimplemented!(),
        "min_heap_size" => unimplemented!(),
        "priority" => unimplemented!(),
        "save_calls" => unimplemented!(),
        "sensitive" => unimplemented!(),
        "trap_exit" => {
            let value_bool: bool = term_try_into_bool("trap_exit value", value)?;

            Ok(process.trap_exit(value_bool).into())
        }
        name => Err(TryAtomFromTermError(name)).context("supported flags are error_handler, max_heap_size, message_queue_data, min_bin_vheap_size, min_heap_size, priority, save_calls, sensitive, and trap_exit").map_err(From::from),
    }
}

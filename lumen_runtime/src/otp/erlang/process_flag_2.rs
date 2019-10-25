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

#[native_implemented_function(process_flag/2)]
pub fn native(process: &Process, flag: Term, value: Term) -> exception::Result {
    let flag_atom: Atom = flag.try_into()?;

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
            let value_bool: bool = value.try_into()?;

            Ok(process.trap_exit(value_bool).into())
        }
        _ => Err(badarg!().into()),
    }
}

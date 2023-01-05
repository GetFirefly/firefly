#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use std::ptr::NonNull;

use anyhow::*;

use firefly_rt::error::ErlangException;
use firefly_rt::process::Process;
use firefly_rt::term::Term;

use crate::runtime::context::*;

#[native_implemented::function(erlang:process_flag/2)]
pub fn result(
    process: &Process,
    flag: Term,
    value: Term,
) -> Result<Term, NonNull<ErlangException>> {
    let flag_atom = term_try_into_atom!(flag)?;

    match flag_atom.as_str() {
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

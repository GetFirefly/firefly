// wasm32 proptest cannot be compiled at the same time as non-wasm32 proptest, so disable tests that
// use proptest completely for wasm32
//
// See https://github.com/rust-lang/cargo/issues/4866
#[cfg(all(not(target_arch = "wasm32"), test))]
mod test;

use crate::time::datetime;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;
use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(universaltime/0)]
pub fn native(process: &Process) -> exception::Result {
    let now = datetime::utc_now();
    let date = now.0;
    let time = now.1;

    let date_tuple = process.tuple_from_iter(vec![date.0, date.1, date.2].iter(), 3);
    let time_tuple = process.tuple_from_iter(vec![time.0, time.1, time.2].iter(), 3);
    Ok(process.tuple_from_iter(vec![date_tuple, time_tuple].iter(), 2));
}

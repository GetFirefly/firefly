//! Generates an integer between 0 and (exclusive_max - 1).
//!
//! ```elixir
//! random_integer = Lumen.Web.Math.random_integer(exclusive_max)
//! ```

use std::convert::TryInto;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::Term;

use lumen_runtime_macros::native_implemented_function;

#[native_implemented_function(random_integer/1)]
fn native(process: &Process, exclusive_max: Term) -> exception::Result {
    let exclusive_max_usize: usize = exclusive_max.try_into()?;
    let exclusive_max_f64 = exclusive_max_usize as f64;
    let random_usize = (js_sys::Math::random() * exclusive_max_f64).trunc() as usize;

    process.integer(random_usize).map_err(|error| error.into())
}

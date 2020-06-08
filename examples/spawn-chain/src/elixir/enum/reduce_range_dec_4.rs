//! ```elixir
//! defp reduce_range_dec(first, first, acc, fun) do
//!   fun.(first, acc)
//! end
//!
//! defp reduce_range_dec(first, last, acc, fun) do
//!   reduce_range_dec(first - 1, last, fun.(first, acc), fun)
//! end
//! ```

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

#[native_implemented::function(reduce_range_dec/4)]
fn result(_process: &Process, _first: Term, _last: Term, _acc: Term, _reducer: Term) -> Term {
    unimplemented!()
}

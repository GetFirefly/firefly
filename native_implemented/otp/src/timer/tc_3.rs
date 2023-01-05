//! ```elixir
//! def tc(module, function, arguments) do
//!   before = :erlang.monotonic_time()
//!   value = apply(module, function, arguments)
//!   after = :erlang.monotonic_time()
//!   duration = after - before
//!   time = :erlang.convert_time_unit(duration, :native, :microsecond)
//!   {time, value}
//! end
//! ```

use firefly_rt::process::Process;
use firefly_rt::term::{Atom, Term};

use crate::erlang::apply_3::apply_3;
use crate::erlang::convert_time_unit_3;
use crate::erlang::monotonic_time_0;

// Private

#[native_implemented::function(timer:tc/3)]
async fn result(process: &Process, module: Term, function: Term, arguments: Term) -> Term {
    let before = monotonic_time_0::result(process).await;
    let value = apply_3(module, function, arguments).await;
    let after = monotonic_time_0::result(process).await;
    let duration = after - before;
    let time = convert_time_unit_3::result(process, duration.into(),  Atom::str_to_term("native").into(),
                                           Atom::str_to_term("microsecond").into()).await?;

    process.tuple_term_from_term_slice(&[time, value])
}

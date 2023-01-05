//! ```elixir
//! case Lumen.Web.Window.window() do
//!    {:ok, window} -> ...
//!    :error -> ...
//! end
//! ```

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::option_to_ok_tuple_or_error;

#[native_implemented::function(Elixir.Lumen.Web.Window:window/0)]
pub fn result(process: &Process) -> Term {
    let option_window = web_sys::window();

    option_to_ok_tuple_or_error(process, option_window)
}

//! ```elixir
//! case Lumen.Web.Window.document(window) do
//!    {:ok, document} -> ...
//!    :error -> ...
//! end
//! ```

use std::convert::TryInto;

use web_sys::Window;

use liblumen_alloc::badarg;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{resource, Term};

use lumen_runtime_macros::native_implemented_function;

use crate::option_to_ok_tuple_or_error;

#[native_implemented_function(document/1)]
pub fn native(process: &Process, window: Term) -> exception::Result {
    let window_reference: resource::Reference = window.try_into()?;
    let window_window: &Window = window_reference.downcast_ref().ok_or_else(|| badarg!())?;
    let option_document = window_window.document();

    option_to_ok_tuple_or_error(process, option_document).map_err(|error| error.into())
}

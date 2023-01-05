//! ```elixir
//! case Lumen.Web.Window.document(window) do
//!    {:ok, document} -> ...
//!    :error -> ...
//! end
//! ```

use std::convert::TryInto;

use anyhow::*;
use web_sys::Window;

use liblumen_alloc::erts::exception;
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::option_to_ok_tuple_or_error;

#[native_implemented::function(Elixir.Lumen.Web.Window:document/1)]
pub fn result(process: &Process, window: Term) -> exception::Result<Term> {
    let boxed: Boxed<Resource> = window
        .try_into()
        .with_context(|| format!("window ({}) must be a window resource", window))?;
    let window_reference: Resource = boxed.into();
    let window_window: &Window = window_reference
        .downcast_ref()
        .with_context(|| format!("window ({}) is a resource, but not a window", window))?;
    let option_document = window_window.document();

    Ok(option_to_ok_tuple_or_error(process, option_document))
}

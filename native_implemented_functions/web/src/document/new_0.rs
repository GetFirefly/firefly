//! Returns a new document whose `origin` is the `origin` of the current global object's associated
//! `Document`.
//!
//! ```elixir
//! case Lumen.Web.Document.new() do
//!    {:ok, document} -> ...
//!    :error -> ...
//! end
//! ```

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use native_implemented_function::native_implemented_function;

use crate::ok_tuple;

#[native_implemented_function(new/0)]
fn result(process: &Process) -> exception::Result<Term> {
    match web_sys::Document::new() {
        Ok(document) => ok_tuple(process, Box::new(document)).map_err(|error| error.into()),
        // Not sure how this can happen
        Err(_) => Ok(atom!("error")),
    }
}

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
use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::Term;

use crate::ok_tuple;

#[native_implemented::function(Elixir.Lumen.Web.Document:new/0)]
fn result(process: &Process) -> Term {
    match web_sys::Document::new() {
        Ok(document) => ok_tuple(process, document),
        // Not sure how this can happen
        Err(_) => atom!("error"),
    }
}

//! ```elixir
//! {:ok, document} = Lumen.Web.Document.new()
//! body_tuple = Lumen.Web.Document.body(document)
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

#[path = "without_body/label_1.rs"]
pub mod label_1;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::document;

#[native_implemented::function(Elixir.Lumen.Web.Document.Body1:without_body/0)]
fn result(process: &Process) -> Term {
    // ```elixir
    // # pushed to stack: ()
    // # returned from call: N/A
    // # full stack: ()
    // # returns: {:ok, document}
    // ```
    process.queue_frame_with_arguments(document::new_0::frame().with_arguments(false, &[]));
    // ```elixir
    // # label 1
    // # pushed to stack: ()
    // # returned from call: {:ok, document}
    // # full stack: ({:ok, document})
    // # returns: {:ok, body} | :error
    // body_tuple = Lumen.Web.Document.body(document)
    // Lumen.Web.Wait.with_return(body_tuple)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}

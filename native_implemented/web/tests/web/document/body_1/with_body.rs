//! ```elixir
//! {:ok, document} = Lumen.Web.Document.new()
//! body_tuple = Lumen.Web.Document.body(document)
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

#[path = "with_body/label_1.rs"]
pub mod label_1;
#[path = "with_body/label_2.rs"]
pub mod label_2;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::window;

#[native_implemented::function(Elixir.Lumen.Web.Document.Body1:with_body/0)]
fn result(process: &Process) -> Term {
    // ```elixir
    // # pushed to stack: ()
    // # returned from call: N/A
    // # full stack: ()
    // # returns: {:ok, window}
    // ```
    process.queue_frame_with_arguments(window::window_0::frame().with_arguments(false, &[]));
    // ```elixir
    // # label 1
    // # pushed to stack: ()
    // # returned from call: {:ok, window}
    // # full stack: ({:ok, window})
    // # returns: {:ok, document}
    // {:ok, document} = Lumen.Web.Window.document(window)
    // body_tuple = Lumen.Web.Document.body(document)
    // Lumen.Web.Wait.with_return(body_tuple)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}

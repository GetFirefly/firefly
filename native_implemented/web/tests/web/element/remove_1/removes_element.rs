//! ```elixir
//! {:ok, window} = Lumen.Web.Window.window()
//! {:ok, document} = Lumen.Web.Window.document(window)
//! {:ok, body} = Lumen.Web.Document.body(document)
//! {:ok, child} = Lumen.Web.Document.create_element(body, "table");
//! :ok = Lumen.Web.Node.append_child(document, child);
//! :ok = Lumen.Web.Element.remove(child);
//! Lumen.Web.Wait.with_return(body_tuple)
//! ```

#[path = "removes_element/label_1.rs"]
pub mod label_1;
#[path = "removes_element/label_2.rs"]
pub mod label_2;
#[path = "removes_element/label_3.rs"]
pub mod label_3;
#[path = "removes_element/label_4.rs"]
pub mod label_4;
#[path = "removes_element/label_5.rs"]
pub mod label_5;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::window;

#[native_implemented::function(Elixir.Lumen.Web.Element.Remove1:removes_element/0)]
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
    // {:ok, body} = Lumen.Web.Document.body(document)
    // {:ok, child} = Lumen.Web.Document.create_element(body, "table");
    // :ok = Lumen.Web.Node.append_child(body, child);
    // :ok = Lumen.Web.Element.remove(child);
    // Lumen.Web.Wait.with_return(body_tuple)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}

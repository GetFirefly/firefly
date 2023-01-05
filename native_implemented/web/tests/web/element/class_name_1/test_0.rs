//! ```elixir
//! {:ok, window} = Lumen.Web.Window.window()
//! {:ok, document} = Lumen.Web.Window.document(window)
//! {:ok, body} = Lumen.Web.Document.body(document)
//! class_name = Lumen.Web.Element.class_name(body)
//! Lumen.Web.Wait.with_return(class_name)
//! ```

#[path = "test_0/label_1.rs"]
pub mod label_1;
#[path = "test_0/label_2.rs"]
pub mod label_2;
#[path = "test_0/label_3.rs"]
pub mod label_3;

use liblumen_rt::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_web::window;

#[native_implemented::function(Elixir.Lumen.Web.Element.ClassName1:test_0/0)]
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
    // class_name = Lumen.Web.Element.class_name(body)
    // Lumen.Web.Wait.with_return(class_name)
    // ```
    process.queue_frame_with_arguments(label_1::frame().with_arguments(true, &[]));

    Term::NONE
}

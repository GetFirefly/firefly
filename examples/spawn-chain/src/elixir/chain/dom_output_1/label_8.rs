use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::process::frames::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{frames, Process};
use liblumen_alloc::erts::term::prelude::{Boxed, Resource, Term};

use crate::elixir::chain::dom_output_1::label_9;

pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tr: Term,
) -> frames::Result {
    process.stack_push(tr)?;
    process.stack_push(document)?;
    process.place_frame(frame(process), placement);

    Ok(())
}

// Private

/// ```elixir
/// # label 8
/// # pushed to stack: (document, tr)
/// # returned from call: text_text
/// # full stack: (text_text, document, tr)
/// # returns: {:ok, text_td}
/// {:ok, text_td} = Lumen::Web::Document.create_element(document, "td")
/// Lumen::Web::Node.append_child(text_td, text_text)
/// Lumen::Web::Node.append_child(tr, text_td)
///
/// {:ok, tbody} = Lumen::Web::Document.get_element_by_id(document, "output")
/// Lumen::Web::Node.append_child(tbody, tr)
/// ```
fn code(arc_process: &Arc<Process>) -> frames::Result {
    arc_process.reduce();

    let text_text = arc_process.stack_pop().unwrap();
    let _: Boxed<Resource> = text_text.try_into().unwrap();
    let document = arc_process.stack_pop().unwrap();
    let _: Boxed<Resource> = document.try_into().unwrap();
    let tr = arc_process.stack_pop().unwrap();
    let _: Boxed<Resource> = tr.try_into().unwrap();

    label_9::place_frame_with_arguments(arc_process, Placement::Replace, document, tr, text_text)
        .unwrap();

    let tag = arc_process.binary_from_str("td").unwrap();
    liblumen_web::document::create_element_2::place_frame_with_arguments(
        arc_process,
        Placement::Push,
        document,
        tag,
    )
    .unwrap();

    Process::call_native_or_yield(arc_process)
}

fn frame(process: &Process) -> Frame {
    let module_function_arity = process.current_module_function_arity().unwrap();

    Frame::new(module_function_arity, code)
}

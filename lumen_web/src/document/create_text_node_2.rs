use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::document::document_from_term;
use lumen_runtime::binary_to_string::binary_to_string;

/// ```elixir
/// text = Lumen.Web.Document.create_text_node(document, data)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    data: Term,
) -> Result<(), Alloc> {
    process.stack_push(data)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let document = arc_process.stack_pop().unwrap();
    let data = arc_process.stack_pop().unwrap();

    match native(arc_process, document, data) {
        Ok(ok_or_error) => {
            arc_process.return_from_call(ok_or_error)?;

            Process::call_code(arc_process)
        }
        Err(exception) => result_from_exception(arc_process, exception),
    }
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("create_text_node").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub fn native(process: &Process, document: Term, data: Term) -> exception::Result<Term> {
    let document_document = document_from_term(document)?;
    let data_string: String = binary_to_string(data)?;

    let text = document_document.create_text_node(&data_string);
    let text_box = Box::new(text);

    process.resource(text_box).map_err(|error| error.into())
}

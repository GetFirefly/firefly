use std::sync::Arc;

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::binary_to_string::binary_to_string;

use crate::document::document_from_term;
use crate::ok_tuple;

/// ```elixir
/// case Lumen.Web.Document.create_element(document, tag) do
///   {:ok, element} -> ...
///   {:error, {:tag, tag}} -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    document: Term,
    tag: Term,
) -> Result<(), Alloc> {
    process.stack_push(tag)?;
    process.stack_push(document)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let document = arc_process.stack_pop().unwrap();
    let tag = arc_process.stack_pop().unwrap();

    match native(arc_process, document, tag) {
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
    Atom::try_from_str("create_element").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 2,
    })
}

pub fn native(process: &Process, document: Term, tag: Term) -> exception::Result<Term> {
    let document_document = document_from_term(document)?;
    let tag_string: String = binary_to_string(tag)?;

    match document_document.create_element(&tag_string) {
        Ok(element) => ok_tuple(process, Box::new(element)),
        Err(_) => {
            let tag_tag = Atom::str_to_term("tag");
            let reason = process.tuple_from_slice(&[tag_tag, tag])?;

            let error = atom!("error");

            process.tuple_from_slice(&[error, reason])
        }
    }
    .map_err(|error| error.into())
}

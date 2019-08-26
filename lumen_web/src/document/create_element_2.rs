use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::code::{self, result_from_exception};
use liblumen_alloc::erts::process::ProcessControlBlock;
use liblumen_alloc::erts::term::{atom_unchecked, Atom, Term};
use liblumen_alloc::erts::ModuleFunctionArity;

use crate::document::document_from_term;
use crate::{error, ok_tuple};

/// ```elixir
/// case Lumen.Web.Document.create_element(document, tag) do
///   {:ok, element} -> ...
///   {:error, {:tag, tag}} -> ...
/// end
/// ```
pub fn place_frame_with_arguments(
    process: &ProcessControlBlock,
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

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    arc_process.reduce();

    let document = arc_process.stack_pop().unwrap();
    let tag = arc_process.stack_pop().unwrap();

    match native(arc_process, document, tag) {
        Ok(ok_or_error) => {
            arc_process.return_from_call(ok_or_error)?;

            ProcessControlBlock::call_code(arc_process)
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

pub fn native(process: &ProcessControlBlock, document: Term, tag: Term) -> exception::Result {
    let document_document = document_from_term(document)?;

    let tag_string: String = tag.try_into()?;

    match document_document.create_element(&tag_string) {
        Ok(element) => ok_tuple(process, Box::new(element)),
        Err(_) => {
            let tag_tag = atom_unchecked("tag");
            let reason = process.tuple_from_slice(&[tag_tag, tag])?;

            let error = error();

            process.tuple_from_slice(&[error, reason])
        }
    }
    .map_err(|error| error.into())
}

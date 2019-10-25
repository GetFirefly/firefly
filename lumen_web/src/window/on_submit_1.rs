use std::convert::TryInto;
use std::sync::Arc;

use wasm_bindgen::JsCast;

use web_sys::{Event, HtmlFormElement};

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::code::stack::frame::{Frame, Placement};
use liblumen_alloc::erts::process::{code, Process};
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::ModuleFunctionArity;

use lumen_runtime::otp::erlang;

use crate::error;

/// The global `onsubmit` event listener for Lumen.Web forms.
///
/// Automatically registered on start.
///
/// ```elixir
/// :ok = Lumen.Web.Window.on_submit(event)
/// ```
pub fn place_frame_with_arguments(
    process: &Process,
    placement: Placement,
    event: Term,
) -> Result<(), Alloc> {
    process.stack_push(event)?;
    process.place_frame(frame(), placement);

    Ok(())
}

// Private

fn code(arc_process: &Arc<Process>) -> code::Result {
    arc_process.reduce();

    let event = arc_process.stack_pop().unwrap();

    // `.unwrap` on both of these because `on_submit_1` should only be called by code controlled
    // by us and it is a bug in `lumen_web` if these don't succeed
    let event_reference: resource::Reference = event.try_into().unwrap();
    let event_event: &Event = event_reference.downcast_ref().unwrap();

    if let Some(event_target) = event_event.target() {
        let result_html_form_element: Result<HtmlFormElement, _> = event_target.dyn_into();

        match result_html_form_element {
            Ok(html_form_element) => {
                match html_form_element.get_attribute("data-lumen-submit-module") {
                    Some(lumen_submit_module_string) => {
                        match html_form_element.get_attribute("data-lumen-submit-function") {
                            Some(lumen_submit_function_string) => {
                                let module = Atom::str_to_term(&lumen_submit_module_string);
                                let function = Atom::str_to_term(&lumen_submit_function_string);
                                let arguments = arc_process.list_from_slice(&[event])?;

                                erlang::apply_3::place_frame_with_arguments(
                                    arc_process,
                                    Placement::Replace,
                                    module,
                                    function,
                                    arguments,
                                )?;
                            }
                            None => {
                                let error_tuple = arc_process.tuple_from_slice(&[
                                    error(),
                                    Atom::str_to_term("data-lumen-submit-function"),
                                ])?;
                                arc_process.return_from_call(error_tuple)?;
                            }
                        }
                    }
                    None => {
                        // A form not being managed by lumen, so ignore
                        arc_process.return_from_call(Atom::str_to_term("ignore"))?;
                    }
                }
            }
            Err(_) => {
                // Only form submission is supported at this time, so ignore
                arc_process.return_from_call(Atom::str_to_term("ignore"))?;
            }
        }
    }

    Process::call_code(arc_process)
}

fn frame() -> Frame {
    Frame::new(module_function_arity(), code)
}

fn function() -> Atom {
    Atom::try_from_str("on_submit").unwrap()
}

fn module_function_arity() -> Arc<ModuleFunctionArity> {
    Arc::new(ModuleFunctionArity {
        module: super::module(),
        function: function(),
        arity: 1,
    })
}

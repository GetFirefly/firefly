//! The global `onsubmit` event listener for Lumen.Web forms.
//!
//! Automatically registered on start.
//!
//! ```elixir
//! :ok = Lumen.Web.Window.on_submit(event)
//! ```

use std::convert::TryInto;

use wasm_bindgen::JsCast;

use web_sys::{Event, HtmlFormElement};

use liblumen_alloc::atom;
use liblumen_alloc::erts::exception;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use liblumen_otp::erlang;

use native_implemented_function::native_implemented_function;

#[native_implemented_function(on_submit/1)]
fn result(process: &Process, event: Term) -> exception::Result<Term> {
    // `.unwrap` on both of these because `on_submit_1` should only be called by code controlled
    // by us and it is a bug in `lumen_web` if these don't succeed
    let boxed: Boxed<Resource> = event.try_into().unwrap();
    let event_reference: Resource = boxed.into();
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
                                let arguments = process.list_from_slice(&[event])?;

                                let frame_with_arguments = erlang::apply_3::frame_with_arguments(
                                    module, function, arguments,
                                );
                                process.queue_frame_with_arguments(frame_with_arguments);

                                Ok(Term::NONE)
                            }
                            None => process
                                .tuple_from_slice(&[
                                    atom!("error"),
                                    atom!("data-lumen-submit-function"),
                                ])
                                .map_err(From::from),
                        }
                    }
                    None => {
                        // A form not being managed by lumen, so ignore
                        ignore()
                    }
                }
            }
            Err(_) => {
                // Only form submission is supported at this time, so ignore
                ignore()
            }
        }
    } else {
        ignore()
    }
}

fn ignore() -> exception::Result<Term> {
    Ok(Atom::str_to_term("ignore"))
}

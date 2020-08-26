use crate::erts::process::Frame;
use crate::erts::term::prelude::*;

#[derive(Debug)]
pub struct FrameWithArguments {
    pub frame: Frame,
    pub uses_returned: bool,
    pub arguments: Vec<Term>,
}

impl FrameWithArguments {
    pub fn new(frame: Frame, uses_returned: bool, arguments: &[Term]) -> Self {
        let native = frame.native();
        let native_arity = native.arity() as usize;
        let returned_len = if uses_returned { 1 } else { 0 };
        let arguments_len = arguments.len();
        let total_arguments_len = returned_len + arguments_len;

        if (frame.native().arity() as usize) != total_arguments_len {
            let joined_formatted_elements = arguments
                .iter()
                .map(Term::to_string)
                .collect::<Vec<String>>()
                .join(", ");
            let formatted_arguments = format!("[{}]", joined_formatted_elements);

            panic!(
                "{} returned plus arguments ({}) length ({}) does not match arity ({}) of native ({:?}) in frame ({:?})",
                if uses_returned { "With" } else { "Without" },
                formatted_arguments,
                total_arguments_len,
                native_arity,
                native,
                frame
            );
        }

        Self {
            frame,
            uses_returned,
            arguments: arguments.to_vec(),
        }
    }
}

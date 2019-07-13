use std::sync::Arc;

pub fn puts_frame() -> Frame {
    let module_function_arity = Arc::new(ModuleFunctionArity {
        module: Term::str_to_atom("Elixir.IO", DoNotCare).unwrap(),
        function: Term::str_to_atom("puts", DoNotCare).unwrap(),
        arity: 1,
    });

    Frame::new(module_function_arity, puts_code)
}

fn puts_code(arc_process: &Arc<Process>) {
    let frame_argument_vec = arc_process.pop_arguments(1);
    let elixir_string = frame_argument_vec[0];

    match elixir_string.try_into(): Result<String, Exception> {
        Ok(string) => {
            // NOT A DEBUGGING LOG
            crate::start::log_1(string);
            arc_process.reduce();

            let ok = Term::str_to_atom("ok", DoNotCare).unwrap();
            arc_process.return_from_call(ok);

            Process::call_code(arc_process);
        }
        Err(exception) => {
            arc_process.reduce();
            arc_process.exception(exception);
        }
    }
}

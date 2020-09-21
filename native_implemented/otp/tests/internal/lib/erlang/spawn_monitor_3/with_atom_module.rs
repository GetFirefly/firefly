#[path = "with_atom_module/with_atom_function.rs"]
mod with_atom_function;

test_stdout!(
    without_atom_function_errors_badarg,
    "{caught, error, badarg}\n"
);

#[path = "with_atom_module/with_atom_function.rs"]
mod with_atom_function;

test_stdout_substrings!(
    without_atom_function_errors_badarg,
    vec!["{caught, error, badarg}\n"]
);

#[path = "with_atom_function/with_empty_list_arguments.rs"]
mod with_empty_list_arguments;
#[path = "with_atom_function/with_non_empty_proper_list_arguments.rs"]
mod with_non_empty_proper_list_arguments;

test_stdout!(
    without_proper_list_arguments_errors_badarg,
    "{caught, error, badarg}\n"
);

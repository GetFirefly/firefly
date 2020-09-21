#[path = "with_exported_function/with_arity.rs"]
mod with_arity;

test_substrings!(
    without_arity_when_run_exits_undef_and_parent_exits,
    vec!["{parent, exited, undef}"],
    vec!["Process (#PID<0.3.0>) exited abnormally.", "undef"]
);

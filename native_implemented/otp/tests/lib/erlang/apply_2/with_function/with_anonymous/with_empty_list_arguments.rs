test_stdout!(
    without_arity_errors_badarity,
    "{caught, error, badarity, with, args, []}\n"
);
test_stdout!(
    with_arity_returns_function_return,
    "from_fun\nfrom_environment\n"
);

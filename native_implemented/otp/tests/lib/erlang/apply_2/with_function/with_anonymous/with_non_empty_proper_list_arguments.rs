test_stdout!(
    without_arity_errors_badarity,
    "{caught, error, badarity, with, args, [0]}\n"
);
test_stdout!(
    with_arity_returns_function_return,
    "from_fun\nfrom_environment\n[argument_a, argument_b]\n[from_environment, argument_a, argument_b]\n"
);

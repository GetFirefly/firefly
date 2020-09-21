test_stdout!(
    without_proper_list_errors_badarg,
    "{caught, error, badarg}\n"
);
test_stdout!(
    without_tuple_list_errors_badarg,
    "{caught, error, badarg}\n"
);
test_stdout!(
    with_two_element_tuple_list_returns_value,
    "#{key => value}\n"
);
test_stdout!(
    with_duplicate_keys_preserves_last_value,
    "#{key => last_value}\n"
);

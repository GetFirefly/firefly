test_stdout!(
    with_key_returns_value_and_removes_key_from_dictionary,
    "value\nundefined\n"
);
test_stdout!(without_key_returns_undefined, "undefined\n");

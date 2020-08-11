// `with_safe` in unit tests

test_stdout!(
    with_used_with_binary_returns_how_many_bytes_were_consumed_along_with_term,
    "{hello, 9}\n{hello, 9}\ntrue\nhello\n"
);

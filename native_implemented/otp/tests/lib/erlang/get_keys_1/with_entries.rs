test_stdout!(without_value_returns_empty_list, "[]\n");
test_stdout!(
    with_value_returns_keys_with_value_in_list,
    "true\ntrue\nfalse\n"
);
// From https://github.com/erlang/otp/blob/a62aed81c56c724f7dd7040adecaa28a78e5d37f/erts/doc/src/erlang.xml#L2104-L2112
test_stdout!(doc_test, "true\ntrue\ntrue\ntrue\ntrue\ntrue\n");

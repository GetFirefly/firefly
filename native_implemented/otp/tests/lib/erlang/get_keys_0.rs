test_stdout!(with_entries_returns_list, "[key]\n");
// From https://github.com/erlang/otp/blob/a62aed81c56c724f7dd7040adecaa28a78e5d37f/erts/doc/src/erlang.xml#L2089-L2094
test_stdout!(doc_test, "true\ntrue\ntrue\n");
test_stdout!(without_entries_returns_empty_list, "[]\n");

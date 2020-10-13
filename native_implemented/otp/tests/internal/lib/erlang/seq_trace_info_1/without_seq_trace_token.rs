test_stdout!(without_atom_errors_badarg, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(
    without_supported_atom_errors_badarg,
    "{caught, error, badarg}\n"
);
test_stdout!(with_label_returns_empty_list, "{label, []}\n");
test_stdout!(
    with_monotonic_timestamp_returns_false,
    "{monotonic_timestamp, false}\n"
);
test_stdout!(with_print_returns_false, "{print, false}\n");
test_stdout!(with_receive_returns_false, "{receive, false}\n");
test_stdout!(with_send_returns_false, "{send, false}\n");
test_stdout!(with_serial_returns_empty_list, "{serial, []}\n");
test_stdout!(with_spawn_returns_false, "{spawn, false}\n");
test_stdout!(
    with_strict_monotonic_timestamp_returns_false,
    "{strict_monotonic_timestamp, false}\n"
);
test_stdout!(with_timestamp_returns_false, "{timestamp, false}\n");

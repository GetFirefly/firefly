// `without_number_errors_badarg` in unit tests

test_stdout!(with_atom, ":'atom'\n");
test_stdout!(with_small_integer, "1\n0\n-1\n");
test_stdout!(with_float, "1.2\n0.3\n-4.5\n");

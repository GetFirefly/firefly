test_stdout!(doctest, "<<169,138,199>>\n");
test_stdout!(with_smallest_big_int, "<<64,0,0,0,0,0>>\n");
test_stdout!(with_non_integer, "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n");
test_stdout!(with_negative_integer, "{caught, error, badarg}\n{caught, error, badarg}\n");

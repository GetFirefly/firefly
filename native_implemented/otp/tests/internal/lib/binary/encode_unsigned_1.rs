test_stdout!(doctest, "<<169,138,199>>\n");
test_stdout!(with_smallest_big_int, "<<64,0,0,0,0,0>>\n");
test_stdout!(
    with_non_integer,
    "{caught, error, badarg}\n{caught, error, badarg}\n{caught, error, badarg}\n"
);
test_stdout!(
    with_negative_integer,
    "{caught, error, badarg}\n{caught, error, badarg}\n"
);
test_stdout!(
    when_big_int_encoded_bytes_have_significant_trailing_zeros,
    "<<64,0,0,0,0,0>>\n"
);
test_stdout!(
    when_small_int_encoded_bytes_have_significant_trailing_zeros,
    "<<1,0,0,0>>\n"
);
test_stdout!(
    when_small_int_encoded_bytes_have_zeros_in_the_middle,
    "<<169,0,199>>\n"
);
test_stdout!(
    when_big_int_encoded_bytes_have_zeros_in_the_middle,
    "<<64,0,0,0,0,1>>\n"
);

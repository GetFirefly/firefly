test_stdout!(
    with_20_digits_is_the_same_as_float_to_list_1,
    "\"0.00000000000000000000e+00\"\n\"0.00000000000000000000e+00\"\n\"1.00000000000000005551e-01\"\n\"1.00000000000000005551e-01\"\n"
);
test_stdout!(returns_list_with_coefficient_e_exponent, "\"1e+09\"\n\"1.2e+09\"\n\"1.23e+09\"\n\"1.235e+09\"\n\"1.2346e+09\"\n\"1.23457e+09\"\n\"1.234568e+09\"\n\"1.2345679e+09\"\n\"1.23456789e+09\"\n\"1.234567890e+09\"\n\"1.2345678901e+09\"\n");
// `always_includes_e` in unit tests
// `always_includes_sign_of_exponent` in unit tests
// `exponent_is_at_least_2_digits` in unit tests

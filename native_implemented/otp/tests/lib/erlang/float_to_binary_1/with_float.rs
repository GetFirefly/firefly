test_stdout!(
    with_float_returns_binary,
    "<<\"-1.19999999999999995559e+00\">>\n<<\"2.99999999999999988898e-01\">>\n<<\"4.50000000000000000000e+00\">>\n"
);
test_stdout!(
    is_the_same_as_float_to_binary_2_with_scientific_20,
    "<<\"0.00000000000000000000e+00\">>\n<<\"1.00000000000000005551e-01\">>\n"
);

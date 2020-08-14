// Only 0-10 to work around https://github.com/lumen/lumen/issues/512
test_stdout!(trailing_zeros_are_truncated, "\"12346\"\n\"12345.7\"\n\"12345.68\"\n\"12345.679\"\n\"12345.6789\"\n\"12345.6789\"\n\"12345.6789\"\n\"12345.6789\"\n\"12345.6789\"\n\"12345.6789\"\n\"12345.6789\"\n");
test_stdout!(
    with_no_fractional_part_still_has_zero_after_decimal_point,
    "\"1.0\"\n"
);

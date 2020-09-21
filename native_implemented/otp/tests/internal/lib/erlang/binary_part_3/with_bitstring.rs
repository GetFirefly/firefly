// `with_bit_count` in unit tests because of https://github.com/lumen/lumen/issues/476
#[path = "with_bitstring/without_bit_count.rs"]
pub mod without_bit_count;

// `without_integer_start_without_integer_length_errors_badarg` in unit tests
// `without_integer_start_with_integer_length_errors_badarg` in unit tests
// `with_non_negative_integer_start_without_integer_length_errors_badarg` in unit tests
// `with_negative_start_with_valid_length_errors_badarg` in unit tests
// `with_start_greater_than_size_with_non_negative_length_errors_badarg` in unit tests
// `with_start_less_than_size_with_negative_length_past_start_errors_badarg` in unit tests
// `with_start_less_than_size_with_positive_length_past_end_errors_badarg` in unit tests

test_stdout!(
    with_positive_start_and_negative_length_returns_subbinary,
    "<<0>>\n"
);

use super::*;

use proptest::strategy::Strategy;

mod with_bit_count;
// `without_bit_count` in integration tests
// `without_integer_start_without_integer_length_errors_badarg` in integration tests
// `without_integer_start_with_integer_length_errors_badarg` in integration tests
// `with_non_negative_integer_start_without_integer_length_errors_badarg` in integration tests
// `with_negative_start_with_valid_length_errors_badarg` in integration tests
// `with_start_greater_than_size_with_non_negative_length_errors_badarg` in integration tests
// `with_start_less_than_size_with_negative_length_past_start_errors_badarg` in integration tests
// `with_start_less_than_size_with_positive_length_past_end_errors_badarg` in integration tests
// `with_positive_start_and_negative_length_returns_subbinary` in integration tests

use proptest::strategy::Just;

use crate::erlang::binary_to_term_1::result;
use crate::test::strategy;

#[test]
fn without_binary_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                result(&arc_process, binary),
                format!("binary ({}) is not a binary", binary)
            );

            Ok(())
        },
    );
}

// `with_binary_encoding_atom_returns_atom` in integration tests
// `with_binary_encoding_empty_list_returns_empty_list` in integration tests
// `with_binary_encoding_list_returns_list` in integration tests
// `with_binary_encoding_small_integer_returns_small_integer` in integration tests
// `with_binary_encoding_integer_returns_integer` in integration tests
// `with_binary_encoding_new_float_returns_float` in integration tests
// `with_binary_encoding_small_tuple_returns_tuple` in integration tests
// `with_binary_encoding_byte_list_returns_list` in integration tests
// `with_binary_encoding_binary_returns_binary` in integration tests
// `with_binary_encoding_small_big_integer_returns_big_integer` in integration tests
// `with_binary_encoding_bit_string_returns_subbinary` in integration tests
// `with_binary_encoding_small_atom_utf8_returns_atom` in integration tests

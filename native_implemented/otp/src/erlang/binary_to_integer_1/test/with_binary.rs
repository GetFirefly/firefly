use super::*;

// `with_small_integer_returns_small_integer` in integration tests
// `with_big_integer_returns_big_integer` in integration tests

#[test]
fn with_non_decimal_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::containing_bytes(
                    "FF".as_bytes().to_owned(),
                    arc_process.clone(),
                ),
            )
        },
        |(arc_process, binary)| {
            prop_assert_badarg!(
                result(&arc_process, binary),
                format!("binary ({}) is not base 10", binary)
            );

            Ok(())
        },
    );
}

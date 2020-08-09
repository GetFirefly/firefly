use proptest::strategy::Just;

use crate::erlang::binary_to_list_1::result;
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
            prop_assert_badarg!(result(&arc_process, binary), format!("binary ({})", binary));

            Ok(())
        },
    );
}

// `with_binary_returns_list_of_bytes` in integration tests

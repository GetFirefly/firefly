use super::*;

use proptest::strategy::Strategy;

mod with_tuple_with_arity_2;

#[test]
fn without_tuple_start_length_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_bitstring(arc_process.clone()),
                strategy::term::is_not_tuple(arc_process),
            )
        },
        |(arc_process, binary, start_length)| {
            prop_assert_badarg!(
                result(&arc_process, binary, start_length),
                format!("start_length ({}) is not a tuple", start_length)
            );

            Ok(())
        },
    );
}

#[test]
fn with_tuple_without_arity_2_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_bitstring(arc_process.clone()),
                strategy::term::tuple(arc_process.clone()).prop_filter(
                    "Tuple must not be arity 2",
                    |start_length| {
                        let tuple: Boxed<Tuple> = (*start_length).try_into().unwrap();

                        tuple.len() != 2
                    },
                ),
            )
        },
        |(arc_process, binary, start_length)| {
            prop_assert_badarg!(
                result(&arc_process, binary, start_length),
                format!(
                    "start_length ({}) is a tuple, but not 2-arity",
                    start_length
                )
            );

            Ok(())
        },
    );
}

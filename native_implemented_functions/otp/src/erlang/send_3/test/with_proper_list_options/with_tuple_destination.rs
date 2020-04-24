use super::*;

use proptest::strategy::Strategy;

mod with_arity_2;

#[test]
fn without_arity_2_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::tuple(arc_process.clone()).prop_filter(
                    "Tuple must not be arity 2",
                    |start_length| {
                        let start_length_tuple: Boxed<Tuple> = (*start_length).try_into().unwrap();

                        start_length_tuple.len() != 2
                    },
                ),
                strategy::term(arc_process.clone()),
                valid_options(arc_process.clone()),
            )
        },
        |(arc_process, destination, message, options)| {
            prop_assert_badarg!(
                result(&arc_process, destination, message, options),
                format!("destination ({}) is a tuple, but not 2-arity", destination)
            );

            Ok(())
        },
    );
}

use super::*;

use proptest::strategy::Strategy;

mod with_arity_2;

#[test]
fn without_arity_2_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()).prop_filter(
                        "Tuple must not be arity 2",
                        |start_length| {
                            let start_length_tuple: Boxed<Tuple> =
                                (*start_length).try_into().unwrap();

                            start_length_tuple.len() != 2
                        },
                    ),
                    strategy::term(arc_process.clone()),
                    valid_options(arc_process.clone()),
                ),
                |(destination, message, options)| {
                    prop_assert_eq!(
                        native(&arc_process, destination, message, options),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

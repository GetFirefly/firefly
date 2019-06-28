use super::*;

use proptest::strategy::Strategy;

mod with_tuple_with_arity_2;

#[test]
fn without_tuple_start_length_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::is_not_tuple(arc_process.clone()),
                ),
                |(binary, start_length)| {
                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_tuple_without_arity_2_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::tuple(arc_process.clone())
                        .prop_filter("Tuple must not be arity 2", |start_length| {
                            start_length.len() != 2
                        }),
                ),
                |(binary, start_length)| {
                    prop_assert_eq!(
                        erlang::binary_part_2(binary, start_length, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

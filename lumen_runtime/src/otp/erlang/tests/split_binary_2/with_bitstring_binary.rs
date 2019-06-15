use super::*;

use proptest::strategy::Strategy;

mod with_heap_binary;
mod with_subbinary;

#[test]
fn without_non_negative_integer_position_errors_badarg() {
    with_process_arc(|arc_process| {
        let zero = &0.into_process(&arc_process);

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Position must not be a non-negative integer", |position| {
                            !(position.is_integer() && (zero <= position))
                        }),
                ),
                |(binary, position)| {
                    prop_assert_eq!(
                        erlang::split_binary_2(binary, position, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_zero_position_returns_empty_prefix_and_binary() {
    with_process_arc(|arc_process| {
        let position = 0.into_process(&arc_process);

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        erlang::split_binary_2(binary, position, &arc_process),
                        Ok(Term::slice_to_tuple(
                            &[Term::slice_to_binary(&[], &arc_process), binary],
                            &arc_process
                        ))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

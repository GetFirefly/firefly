use super::*;

mod with_heap_binary;
mod with_subbinary;

#[test]
fn without_non_negative_integer_position_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_bitstring(arc_process.clone()),
                    strategy::term::is_not_non_negative_integer(arc_process.clone()),
                ),
                |(binary, position)| {
                    prop_assert_eq!(
                        native(&arc_process, binary, position),
                        Err(badarg!(&arc_process).into())
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
        let position = arc_process.integer(0).unwrap();

        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_bitstring(arc_process.clone()),
                |binary| {
                    prop_assert_eq!(
                        native(&arc_process, binary, position),
                        Ok(arc_process
                            .tuple_from_slice(&[
                                arc_process.binary_from_bytes(&[]).unwrap(),
                                binary
                            ],)
                            .unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

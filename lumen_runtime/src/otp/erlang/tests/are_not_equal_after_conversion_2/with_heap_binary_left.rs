use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_binary_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::heap(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a binary", |v| !v.is_binary()),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_heap_binary_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::heap(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(operand, operand),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_value_heap_binary_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::byte_vec().prop_map(|byte_vec| {
                    (
                        Term::slice_to_binary(&byte_vec, &arc_process),
                        Term::slice_to_binary(&byte_vec, &arc_process),
                    )
                }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_heap_binary_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::heap(arc_process.clone()),
                    strategy::term::binary::heap(arc_process.clone()),
                )
                    .prop_filter("Heap binaries must be different", |(left, right)| {
                        left != right
                    }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subbinary_right_with_same_bytes_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::is_binary(arc_process.clone()).prop_map(
                    |subbinary_term| {
                        let subbinary: &sub::Binary = subbinary_term.unbox_reference();
                        let heap_binary_byte_vec: Vec<u8> = subbinary.byte_iter().collect();

                        let heap_binary =
                            Term::slice_to_binary(&heap_binary_byte_vec, &arc_process);
                        (heap_binary, subbinary_term)
                    },
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_subbinary_right_with_different_bytes_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::is_binary::is_not_empty(arc_process.clone())
                    .prop_map(|subbinary_term| {
                        let subbinary: &sub::Binary = subbinary_term.unbox_reference();
                        // same size, but different values by inverting
                        let heap_binary_byte_vec: Vec<u8> =
                            subbinary.byte_iter().map(|b| !b).collect();

                        let heap_binary =
                            Term::slice_to_binary(&heap_binary_byte_vec, &arc_process);
                        (heap_binary, subbinary_term)
                    }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_not_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

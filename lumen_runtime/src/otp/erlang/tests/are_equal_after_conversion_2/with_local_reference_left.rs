use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_local_reference_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right cannot be a local reference",
                        |right| {
                            right.tag() != Boxed
                                || right.unbox_reference::<Term>().tag() != LocalReference
                        },
                    ),
                ),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_local_reference_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::local_reference(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(operand, operand),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_local_reference_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::prelude::any::<u64>().prop_map(move |number| {
                    (
                        Term::local_reference(number, &arc_process),
                        Term::local_reference(number + 1, &arc_process),
                    )
                }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        false.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

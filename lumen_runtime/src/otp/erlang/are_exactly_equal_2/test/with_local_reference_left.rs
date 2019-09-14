use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_local_reference_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::local_reference(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right cannot be a local reference", |right| {
                            !right.is_local_reference()
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

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
                    prop_assert_eq!(native(operand, operand), true.into());

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
                        arc_process.reference(number).unwrap(),
                        arc_process.reference(number + 1).unwrap(),
                    )
                }),
                |(left, right)| {
                    prop_assert_eq!(native(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

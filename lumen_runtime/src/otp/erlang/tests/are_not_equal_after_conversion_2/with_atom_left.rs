use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_atom_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::atom(),
                    strategy::term::is_not_atom(arc_process.clone()),
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
fn with_same_atom_returns_false() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(&strategy::term::atom(), |operand| {
            prop_assert_eq!(
                erlang::are_not_equal_after_conversion_2(operand, operand),
                false.into()
            );

            Ok(())
        })
        .unwrap();
}

#[test]
fn with_different_atom_returns_true() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &(strategy::term::atom(), strategy::term::atom())
                .prop_filter("Atoms must be different", |(left, right)| left != right),
            |(left, right)| {
                prop_assert_eq!(
                    erlang::are_not_equal_after_conversion_2(left, right),
                    true.into()
                );

                Ok(())
            },
        )
        .unwrap();
}

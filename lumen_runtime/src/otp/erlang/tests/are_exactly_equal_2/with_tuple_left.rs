use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_list_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be tuple", |v| !v.is_tuple()),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_tuple_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::tuple(arc_process.clone()), |operand| {
                prop_assert_eq!(erlang::are_exactly_equal_2(operand, operand), true.into());

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_same_value_tuple_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &proptest::collection::vec(
                    strategy::term(arc_process.clone()),
                    strategy::size_range(),
                )
                .prop_map(move |vec| {
                    (
                        Term::slice_to_tuple(&vec, &arc_process),
                        Term::slice_to_tuple(&vec, &arc_process),
                    )
                }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_tuple_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::tuple(arc_process.clone()),
                    strategy::term::tuple(arc_process.clone()),
                )
                    .prop_filter("Tuples must be different", |(left, right)| left != right),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

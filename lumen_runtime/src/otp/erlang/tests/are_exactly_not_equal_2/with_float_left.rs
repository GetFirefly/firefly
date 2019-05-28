use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_float_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a float", |v| {
                            v.tag() != Boxed || v.unbox_reference::<Term>().tag() != Float
                        }),
                ),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_same_float_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::float(arc_process.clone()), |operand| {
                prop_assert_eq!(
                    erlang::are_exactly_not_equal_2(operand, operand),
                    false.into()
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_same_value_float_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<f64>()
                    .prop_map(|f| (f.into_process(&arc_process), f.into_process(&arc_process))),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_float_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &&any::<f64>().prop_map(|f| {
                    (
                        f.into_process(&arc_process),
                        (f / 2.0 + 1.0).into_process(&arc_process),
                    )
                }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_not_equal_2(left, right), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

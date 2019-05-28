use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_small_integer_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a small integer or float", |v| {
                            v.tag() != SmallInteger
                        }),
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
fn with_same_small_integer_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small(arc_process.clone()),
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
fn with_same_value_small_integer_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(crate::integer::small::MIN..crate::integer::small::MAX).prop_map(move |i| {
                    (i.into_process(&arc_process), i.into_process(&arc_process))
                }),
                |(left, right)| {
                    prop_assert_eq!(
                        erlang::are_equal_after_conversion_2(left, right),
                        true.into()
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_small_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(crate::integer::small::MIN..crate::integer::small::MAX).prop_map(move |i| {
                    (
                        i.into_process(&arc_process),
                        (i + 1).into_process(&arc_process),
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

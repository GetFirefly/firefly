use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_small_integer_or_float_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term(arc_process.clone()).prop_filter(
                        "Right must not be a small integer or float",
                        |v| {
                            !(v.tag() == SmallInteger
                                || (v.tag() == Boxed && {
                                    let unboxed_tag = v.unbox_reference::<Term>().tag();

                                    unboxed_tag != Float && unboxed_tag != BigInteger
                                }))
                        },
                    ),
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
fn with_same_small_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::small(arc_process.clone()),
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
fn with_same_value_small_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(crate::integer::small::MIN..crate::integer::small::MAX).prop_map(move |i| {
                    (i.into_process(&arc_process), i.into_process(&arc_process))
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
fn with_different_small_integer_right_returns_true() {
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
fn with_same_value_float_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::small_integer_float_integral_i64().prop_map(move |i| {
                    (
                        i.into_process(&arc_process),
                        (i as f64).into_process(&arc_process),
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
fn with_different_value_float_right_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::small_integer_float_integral_i64().prop_map(move |i| {
                    // change float toward zero to ensure it remains in integral range
                    let diff = if i < 0 { 1 } else { -1 };

                    (
                        i.into_process(&arc_process),
                        ((i + diff) as f64).into_process(&arc_process),
                    )
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

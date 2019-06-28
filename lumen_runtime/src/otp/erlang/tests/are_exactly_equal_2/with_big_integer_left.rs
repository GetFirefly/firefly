use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_big_integer_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Right must not be a big integer or float", |v| {
                            v.tag() != Boxed || v.unbox_reference::<Term>().tag() == BigInteger
                        }),
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
fn with_same_big_integer_returns_true() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big(arc_process.clone()),
                |operand| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(operand, operand), true.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_different_big_integer_right_returns_false() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::integer::big(arc_process.clone()),
                )
                    .prop_filter("Right and left must be different", |(left, right)| {
                        left != right
                    }),
                |(left, right)| {
                    prop_assert_eq!(erlang::are_exactly_equal_2(left, right), false.into());

                    Ok(())
                },
            )
            .unwrap();
    });
}

use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn with_small_integer_divisor_returns_small_integer() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small::isize(),
                divisor(),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_eq!(
                result(
                    &arc_process,
                    arc_process.integer(dividend).unwrap(),
                    arc_process.integer(divisor).unwrap(),
                ),
                Ok(arc_process.integer(dividend % divisor).unwrap())
            );

            Ok(())
        },
    );
}

#[test]
fn with_big_integer_divisor_returns_dividend() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::integer::small(arc_process.clone()),
                strategy::term::integer::big(arc_process.clone()),
            )
        },
        |(arc_process, dividend, divisor)| {
            prop_assert_eq!(result(&arc_process, dividend, divisor), Ok(dividend));

            Ok(())
        },
    );
}

fn divisor() -> BoxedStrategy<isize> {
    prop_oneof![
        (SmallInteger::MIN_VALUE..=-1),
        (1..=SmallInteger::MAX_VALUE)
    ]
    .boxed()
}

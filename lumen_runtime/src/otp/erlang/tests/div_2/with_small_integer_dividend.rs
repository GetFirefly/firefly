use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn with_small_integer_divisor_returns_small_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::integer::small::isize(), divisor()),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::div_2(
                            arc_process.integer(dividend),
                            arc_process.integer(divisor),
                            &arc_process
                        ),
                        Ok(arc_process.integer(dividend / divisor))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_big_integer_divisor_returns_zero() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::small(arc_process.clone()),
                    strategy::term::integer::big(arc_process.clone()),
                ),
                |(dividend, divisor)| {
                    prop_assert_eq!(
                        erlang::div_2(dividend, divisor, &arc_process),
                        Ok(arc_process.integer(0))
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn divisor() -> BoxedStrategy<isize> {
    prop_oneof![
        (SmallInteger::MIN_VALUE..=-1),
        (1..=SmallInteger::MAX_VALUE)
    ]
    .boxed()
}

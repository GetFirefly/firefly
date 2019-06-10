use super::*;

use proptest::strategy::Strategy;

mod with_big_integer_integer;
mod with_small_integer_integer;

#[test]
fn without_integer_integer_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    strategy::term::is_integer(arc_process.clone()),
                ),
                |(integer, shift)| {
                    prop_assert_eq!(
                        erlang::bsr_2(integer, shift, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_without_integer_shift_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    strategy::term::is_not_integer(arc_process.clone()),
                ),
                |(integer, shift)| {
                    prop_assert_eq!(
                        erlang::bsr_2(integer, shift, &arc_process),
                        Err(badarith!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_with_zero_shift_returns_same_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_integer(arc_process.clone()),
                |integer| {
                    let shift = 0.into_process(&arc_process);

                    prop_assert_eq!(erlang::bsr_2(integer, shift, &arc_process), Ok(integer));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_integer_with_integer_shift_is_the_same_as_bsl_with_negated_shift() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(strategy::term::is_integer(arc_process.clone()), shift()),
                |(integer, shift)| {
                    let negated_shift = -1 * shift;

                    prop_assert_eq!(
                        erlang::bsr_2(
                            integer,
                            (shift as isize).into_process(&arc_process),
                            &arc_process
                        ),
                        erlang::bsl_2(
                            integer,
                            (negated_shift as isize).into_process(&arc_process),
                            &arc_process
                        )
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn shift() -> BoxedStrategy<i8> {
    // any::<i8> is not symmetric because i8::MIN is -128 while i8::MAX is 127, so make symmetric
    // range
    (-127_i8..=127_i8).boxed()
}

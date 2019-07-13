use super::*;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

#[test]
fn without_number_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_number(arc_process.clone()),
                |number| {
                    prop_assert_eq!(
                        erlang::negate_1(number, &arc_process),
                        Err(badarith!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_returns_integer_of_opposite_sign() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![std::isize::MIN..=-1, 1..=std::isize::MAX]
                    .prop_map(|i| (arc_process.integer(i), i)),
                |(number, i)| {
                    let negated = arc_process.integer(-i);

                    prop_assert_eq!(erlang::negate_1(number, &arc_process), Ok(negated));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_returns_float_of_opposite_sign() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &prop_oneof![std::f64::MIN..=-1.0, 1.0..=std::f64::MAX]
                    .prop_map(|f| (arc_process.float(f).unwrap(), f)),
                |(number, f)| {
                    let negated = arc_process.float(-f).unwrap();

                    prop_assert_eq!(erlang::negate_1(number, &arc_process), Ok(negated));

                    Ok(())
                },
            )
            .unwrap();
    });
}

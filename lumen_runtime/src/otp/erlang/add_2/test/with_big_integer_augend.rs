use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        native(&arc_process, augend, addend),
                        Err(badarith!(&arc_process).into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_zero_small_integer_returns_same_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big(arc_process.clone()),
                |augend| {
                    let addend = 0.into();

                    prop_assert_eq!(native(&arc_process, augend, addend), Ok(augend));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn that_is_positive_with_positive_small_integer_addend_returns_greater_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big::positive(arc_process.clone()),
                    strategy::term::integer::small::positive(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = native(&arc_process, augend, addend);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(augend < sum);
                    prop_assert!(addend < sum);
                    prop_assert!(sum.is_boxed_bigint());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn that_is_positive_with_positive_big_integer_addend_returns_greater_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::integer::big::positive(arc_process.clone()),
                    strategy::term::integer::big::positive(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = native(&arc_process, augend, addend);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(augend < sum);
                    prop_assert!(addend < sum);
                    prop_assert!(sum.is_boxed_bigint());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, process| {
        let addend = process.float(3.0).unwrap();

        let result = native(&process, augend, addend);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert!(sum.is_boxed_float());
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big::negative(arc_process.clone()),
                |augend| {
                    let addend = arc_process.float(std::f64::MIN).unwrap();

                    prop_assert_eq!(
                        native(&arc_process, augend, addend),
                        Ok(arc_process.float(std::f64::MIN).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::integer::big::positive(arc_process.clone()),
                |augend| {
                    let addend = arc_process.float(std::f64::MAX).unwrap();

                    prop_assert_eq!(
                        native(&arc_process, augend, addend),
                        Ok(arc_process.float(std::f64::MAX).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let augend = process.integer(SmallInteger::MAX_VALUE + 1).unwrap();

        assert!(augend.is_boxed_bigint());

        f(augend, &process)
    })
}

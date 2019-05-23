use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    big_integer_term_strategy(arc_process.clone()),
                    term_is_not_number_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Err(badarith!())
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
            .run(&big_integer_term_strategy(arc_process.clone()), |augend| {
                let addend = 0.into_process(&arc_process);

                prop_assert_eq!(erlang::add_2(augend, addend, &arc_process), Ok(augend));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn that_is_positive_with_positive_small_integer_addend_returns_greater_big_integer() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    positive_big_integer_term_strategy(arc_process.clone()),
                    positive_small_integer_term_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = erlang::add_2(augend, addend, &arc_process);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(augend < sum);
                    prop_assert!(addend < sum);
                    prop_assert_eq!(sum.tag(), Boxed);

                    let unboxed_sum: &Term = sum.unbox_reference();

                    prop_assert_eq!(unboxed_sum.tag(), BigInteger);

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
                    positive_big_integer_term_strategy(arc_process.clone()),
                    positive_big_integer_term_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = erlang::add_2(augend, addend, &arc_process);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(augend < sum);
                    prop_assert!(addend < sum);
                    prop_assert_eq!(sum.tag(), Boxed);

                    let unboxed_sum: &Term = sum.unbox_reference();

                    prop_assert_eq!(unboxed_sum.tag(), BigInteger);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with(|augend, process| {
        let addend = 3.0.into_process(&process);

        let result = erlang::add_2(augend, addend, &process);

        assert!(result.is_ok());

        let sum = result.unwrap();

        assert_eq!(sum.tag(), Boxed);

        let unboxed_sum: &Term = sum.unbox_reference();

        assert_eq!(unboxed_sum.tag(), Float);
    })
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &negative_big_integer_term_strategy(arc_process.clone()),
                |augend| {
                    let addend = std::f64::MIN.into_process(&arc_process);

                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Ok(std::f64::MIN.into_process(&arc_process))
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
                &positive_big_integer_term_strategy(arc_process.clone()),
                |augend| {
                    let addend = std::f64::MAX.into_process(&arc_process);

                    prop_assert_eq!(
                        erlang::add_2(augend, addend, &arc_process),
                        Ok(std::f64::MAX.into_process(&arc_process))
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
        let augend: Term = (crate::integer::small::MAX + 1).into_process(&process);

        assert_eq!(augend.tag(), Boxed);

        let unboxed_augend: &Term = augend.unbox_reference();

        assert_eq!(unboxed_augend.tag(), BigInteger);

        f(augend, &process)
    })
}

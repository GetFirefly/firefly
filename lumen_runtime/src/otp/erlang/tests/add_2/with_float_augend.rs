use super::*;

#[test]
fn without_number_addend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    float_term_strategy(arc_process.clone()),
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
fn with_small_integer_addend_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    float_term_strategy(arc_process.clone()),
                    small_integer_term_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = erlang::add_2(augend, addend, &arc_process);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert_eq!(sum.tag(), Boxed);

                    let unboxed_sum: &Term = sum.unbox_reference();

                    prop_assert_eq!(unboxed_sum.tag(), Float);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_big_integer_addend_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    float_term_strategy(arc_process.clone()),
                    big_integer_term_strategy(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = erlang::add_2(augend, addend, &arc_process);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert_eq!(sum.tag(), Boxed);

                    let unboxed_sum: &Term = sum.unbox_reference();

                    prop_assert_eq!(unboxed_sum.tag(), Float);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_without_underflow_or_overflow_returns_float() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<f64>()
                    .prop_flat_map(|augend_f64| {
                        (
                            Just(augend_f64),
                            Float::clamp_inclusive_range(
                                (std::f64::MIN - augend_f64)..=(std::f64::MAX - augend_f64),
                            ),
                        )
                    })
                    .prop_map(|(augend_f64, addend_f64)| {
                        (
                            augend_f64.into_process(&arc_process),
                            addend_f64.into_process(&arc_process),
                        )
                    }),
                |(augend, addend)| {
                    let result = erlang::add_2(augend, addend, &arc_process);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert_eq!(sum.tag(), Boxed);

                    let unboxed_sum: &Term = sum.unbox_reference();

                    prop_assert_eq!(unboxed_sum.tag(), Float);

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with(|augend, process| {
        let addend = std::f64::MIN.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(std::f64::MIN.into_process(&process))
        );
    })
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with(|augend, process| {
        let addend = std::f64::MAX.into_process(&process);

        assert_eq!(
            erlang::add_2(augend, addend, &process),
            Ok(std::f64::MAX.into_process(&process))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let augend = 2.0.into_process(&process);

        f(augend, &process)
    })
}

use super::*;

use proptest::strategy::Strategy;

#[test]
fn without_number_addend_errors_badarith() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    strategy::term::is_not_number(arc_process.clone()),
                ),
                |(augend, addend)| {
                    prop_assert_eq!(
                        native(&arc_process, augend, addend),
                        Err(badarith!().into())
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
                    strategy::term::float(arc_process.clone()),
                    strategy::term::integer::small(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = native(&arc_process, augend, addend);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(sum.is_float());

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
                    strategy::term::float(arc_process.clone()),
                    strategy::term::integer::big(arc_process.clone()),
                ),
                |(augend, addend)| {
                    let result = native(&arc_process, augend, addend);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(sum.is_float());

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
                            arc_process.float(augend_f64).unwrap(),
                            arc_process.float(addend_f64).unwrap(),
                        )
                    }),
                |(augend, addend)| {
                    let result = native(&arc_process, augend, addend);

                    prop_assert!(result.is_ok());

                    let sum = result.unwrap();

                    prop_assert!(sum.is_float());

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_float_addend_with_underflow_returns_min_float() {
    with(|augend, process| {
        let addend = process.float(std::f64::MIN).unwrap();

        assert_eq!(
            native(&process, augend, addend),
            Ok(process.float(std::f64::MIN).unwrap())
        );
    })
}

#[test]
fn with_float_addend_with_overflow_returns_max_float() {
    with(|augend, process| {
        let addend = process.float(std::f64::MAX).unwrap();

        assert_eq!(
            native(&process, augend, addend),
            Ok(process.float(std::f64::MAX).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &ProcessControlBlock) -> (),
{
    with_process(|process| {
        let augend = process.float(2.0).unwrap();

        f(augend, &process)
    })
}

use super::*;

use num_traits::Num;

use proptest::prop_oneof;
use proptest::strategy::Strategy;

use crate::time::Unit;

#[test]
fn without_integer_time_returns_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_integer(arc_process.clone()),
                    unit_term(arc_process.clone()),
                    unit_term(arc_process.clone()),
                ),
                |(time, from_unit, to_unit)| {
                    prop_assert_eq!(
                        erlang::convert_time_unit_3(time, from_unit, to_unit, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_time_without_unit_from_unit_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    is_not_unit_term(arc_process.clone()),
                    unit_term(arc_process.clone()),
                ),
                |(time, from_unit, to_unit)| {
                    prop_assert_eq!(
                        erlang::convert_time_unit_3(time, from_unit, to_unit, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_integer_time_with_unit_from_unit_without_unit_to_unit_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_integer(arc_process.clone()),
                    unit_term(arc_process.clone()),
                    is_not_unit_term(arc_process.clone()),
                ),
                |(time, from_unit, to_unit)| {
                    prop_assert_eq!(
                        erlang::convert_time_unit_3(time, from_unit, to_unit, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_small_integer_time_valid_units_returns_converted_value() {
    with_process(|process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&(from_unit(), to_unit()), |(from_unit, to_unit)| {
                // not using `proptest` for time to allow math to be hard-coded and not copy the
                // code-under-test
                let time = 1_000_000_000.into_process(process);

                let expected_converted = match (&from_unit, &to_unit) {
                    (Hertz(_), Hertz(_)) => 2_000_000_000.into_process(process),
                    (Hertz(_), Second) => 500_000_000.into_process(process),
                    (Hertz(_), Millisecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Microsecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Nanosecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Native) => <BigInt as Num>::from_str_radix("500_000_000_000", 10)
                        .unwrap()
                        .into_process(process),
                    (Hertz(_), PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Second)
                    | (Millisecond, Millisecond)
                    | (Microsecond, Microsecond)
                    | (Nanosecond, Nanosecond)
                    | (Native, Native)
                    | (PerformanceCounter, PerformanceCounter) => time,
                    (Second, Hertz(_)) => <BigInt as Num>::from_str_radix("5_000_000_000", 10)
                        .unwrap()
                        .into_process(process),
                    (Second, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Native) => <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                        .unwrap()
                        .into_process(process),
                    (Second, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Hertz(_)) => 5_000_000.into_process(process),
                    (Millisecond, Second) => 1_000_000.into_process(process),
                    (Millisecond, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Native) => 1_000_000_000.into_process(process),
                    (Millisecond, PerformanceCounter) => 1_000_000_000.into_process(process),
                    (Microsecond, Hertz(_)) => 5_000.into_process(process),
                    (Microsecond, Second) => 1_000.into_process(process),
                    (Microsecond, Millisecond) => 1_000_000.into_process(process),
                    (Microsecond, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Native) => 1_000_000.into_process(process),
                    (Microsecond, PerformanceCounter) => 1_000_000.into_process(process),
                    (Nanosecond, Hertz(_)) => 5.into_process(process),
                    (Nanosecond, Second) => 1.into_process(process),
                    (Nanosecond, Millisecond) => 1_000.into_process(process),
                    (Nanosecond, Microsecond) => 1_000_000.into_process(process),
                    (Nanosecond, Native) => 1_000.into_process(process),
                    (Nanosecond, PerformanceCounter) => 1_000.into_process(process),
                    (Native, Hertz(_)) => 5_000_000.into_process(process),
                    (Native, Second) => 1_000_000.into_process(process),
                    (Native, Millisecond) => 1_000_000_000.into_process(process),
                    (Native, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, PerformanceCounter) => 1_000_000_000.into_process(process),
                    (PerformanceCounter, Hertz(_)) => 5_000_000.into_process(process),
                    (PerformanceCounter, Second) => 1_000_000.into_process(process),
                    (PerformanceCounter, Millisecond) => 1_000_000_000.into_process(process),
                    (PerformanceCounter, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Native) => 1_000_000_000.into_process(process),
                };

                prop_assert_eq!(
                    erlang::convert_time_unit_3(
                        time,
                        from_unit.into_process(process),
                        to_unit.into_process(process),
                        process
                    ),
                    Ok(expected_converted)
                );

                Ok(())
            })
            .unwrap();
    });
}

// does not use `proptest` so that math can be hard-coded and not match code-under-test
#[test]
fn with_big_integer_time_with_unit_from_unit_with_unit_to_unit_returns_converted_value() {
    with_process(|process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&(from_unit(), to_unit()), |(from_unit, to_unit)| {
                // not using `proptest` for time to allow math to be hard-coded and not copy the
                // code-under-test
                let time = <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                    .unwrap()
                    .into_process(&process);

                let expected_converted = match (&from_unit, &to_unit) {
                    (Hertz(_), Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&process)
                    }
                    (Hertz(_), Second) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Millisecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Microsecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Nanosecond) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), Native) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Hertz(_), PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Second)
                    | (Millisecond, Millisecond)
                    | (Microsecond, Microsecond)
                    | (Nanosecond, Nanosecond)
                    | (Native, Native)
                    | (PerformanceCounter, PerformanceCounter) => time,
                    (Second, Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, Native) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Second, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Second) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(&process)
                    }
                    (Millisecond, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, Native) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Millisecond, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("5_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Second) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, Native) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Microsecond, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Nanosecond, Hertz(_)) => <BigInt as Num>::from_str_radix("5_000_000_000", 10)
                        .unwrap()
                        .into_process(process),
                    (Nanosecond, Second) => <BigInt as Num>::from_str_radix("1_000_000_000", 10)
                        .unwrap()
                        .into_process(process),
                    (Nanosecond, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Nanosecond, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Nanosecond, Native) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Nanosecond, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Second) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (Native, PerformanceCounter) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Hertz(_)) => {
                        <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Second) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Millisecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Microsecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Nanosecond) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                    (PerformanceCounter, Native) => {
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                            .unwrap()
                            .into_process(process)
                    }
                };

                prop_assert_eq!(
                    erlang::convert_time_unit_3(
                        time,
                        from_unit.into_process(process),
                        to_unit.into_process(process),
                        process
                    ),
                    Ok(expected_converted)
                );

                Ok(())
            })
            .unwrap();
    });
}

fn from_unit() -> BoxedStrategy<Unit> {
    prop_oneof![
        // not using `proptest` for time to allow math to be hard-coded and not copy the
        // code-under-test
        Just(Hertz(2)),
        Just(Second),
        Just(Millisecond),
        Just(Microsecond),
        Just(Nanosecond),
        Just(Native),
        Just(PerformanceCounter)
    ]
    .boxed()
}

fn hertz() -> BoxedStrategy<Unit> {
    (1..=std::usize::MAX).prop_map(|hertz| Hertz(hertz)).boxed()
}

fn is_not_unit_term(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term::is_not_integer(arc_process)
        .prop_filter("Term must not be a unit name", |term| match term.tag() {
            Atom => match unsafe { term.atom_to_string() }.as_ref().as_ref() {
                "second" | "millisecond" | "microsecond" | "nanosecond" | "native"
                | "perf_counter" => false,
                _ => true,
            },
            _ => true,
        })
        .boxed()
}

fn to_unit() -> BoxedStrategy<Unit> {
    prop_oneof![
        // not using `proptest` for time to allow math to be hard-coded and not copy the
        // code-under-test
        Just(Hertz(5)),
        Just(Second),
        Just(Millisecond),
        Just(Microsecond),
        Just(Nanosecond),
        Just(Native),
        Just(PerformanceCounter)
    ]
    .boxed()
}

fn unit() -> BoxedStrategy<Unit> {
    prop_oneof![
        hertz(),
        Just(Second),
        Just(Millisecond),
        Just(Microsecond),
        Just(Nanosecond),
        Just(Native),
        Just(PerformanceCounter)
    ]
    .boxed()
}

fn unit_term(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    unit()
        .prop_map(move |unit| unit.into_process(&arc_process))
        .boxed()
}

use std::sync::Arc;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert_eq, prop_oneof};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::convert_time_unit_3::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;
use crate::time::Unit::{self, *};

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
                        native(&arc_process, time, from_unit, to_unit),
                        Err(badarg!(&arc_process).into())
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
                        native(&arc_process, time, from_unit, to_unit),
                        Err(badarg!(&arc_process).into())
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
                        native(&arc_process, time, from_unit, to_unit),
                        Err(badarg!(&arc_process).into())
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
                let time = process.integer(1_000_000_000).unwrap();

                let expected_converted = match (&from_unit, &to_unit) {
                    (Hertz(_), Hertz(_)) => process.integer(2_000_000_000).unwrap(),
                    (Hertz(_), Second) => process.integer(500_000_000).unwrap(),
                    (Hertz(_), Millisecond) => process
                        .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Hertz(_), Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Native) => process
                        .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Hertz(_), PerformanceCounter) => process
                        .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Second, Second)
                    | (Millisecond, Millisecond)
                    | (Microsecond, Microsecond)
                    | (Nanosecond, Nanosecond)
                    | (Native, Native)
                    | (PerformanceCounter, PerformanceCounter) => time,
                    (Second, Hertz(_)) => process
                        .integer(<BigInt as Num>::from_str_radix("5_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Second, Millisecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Second, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Second, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Second, Native) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Second, PerformanceCounter) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Millisecond, Hertz(_)) => process.integer(5_000_000).unwrap(),
                    (Millisecond, Second) => process.integer(1_000_000).unwrap(),
                    (Millisecond, Microsecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Millisecond, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Native) => process.integer(1_000_000_000).unwrap(),
                    (Millisecond, PerformanceCounter) => process.integer(1_000_000_000).unwrap(),
                    (Microsecond, Hertz(_)) => process.integer(5_000).unwrap(),
                    (Microsecond, Second) => process.integer(1_000).unwrap(),
                    (Microsecond, Millisecond) => process.integer(1_000_000).unwrap(),
                    (Microsecond, Nanosecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Microsecond, Native) => process.integer(1_000_000).unwrap(),
                    (Microsecond, PerformanceCounter) => process.integer(1_000_000).unwrap(),
                    (Nanosecond, Hertz(_)) => process.integer(5).unwrap(),
                    (Nanosecond, Second) => process.integer(1).unwrap(),
                    (Nanosecond, Millisecond) => process.integer(1_000).unwrap(),
                    (Nanosecond, Microsecond) => process.integer(1_000_000).unwrap(),
                    (Nanosecond, Native) => process.integer(1_000).unwrap(),
                    (Nanosecond, PerformanceCounter) => process.integer(1_000).unwrap(),
                    (Native, Hertz(_)) => process.integer(5_000_000).unwrap(),
                    (Native, Second) => process.integer(1_000_000).unwrap(),
                    (Native, Millisecond) => process.integer(1_000_000_000).unwrap(),
                    (Native, Microsecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Native, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Native, PerformanceCounter) => process.integer(1_000_000_000).unwrap(),
                    (PerformanceCounter, Hertz(_)) => process.integer(5_000_000).unwrap(),
                    (PerformanceCounter, Second) => process.integer(1_000_000).unwrap(),
                    (PerformanceCounter, Millisecond) => process.integer(1_000_000_000).unwrap(),
                    (PerformanceCounter, Microsecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (PerformanceCounter, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Native) => process.integer(1_000_000_000).unwrap(),
                };

                prop_assert_eq!(
                    native(
                        process,
                        time,
                        from_unit.to_term(process).unwrap(),
                        to_unit.to_term(process).unwrap(),
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
                let time = process
                    .integer(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10).unwrap(),
                    )
                    .unwrap();

                let expected_converted = match (&from_unit, &to_unit) {
                    (Hertz(_), Hertz(_)) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("2_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Second) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Millisecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "500_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), Native) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Hertz(_), PerformanceCounter) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("500_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Second, Second)
                    | (Millisecond, Millisecond)
                    | (Microsecond, Microsecond)
                    | (Nanosecond, Nanosecond)
                    | (Native, Native)
                    | (PerformanceCounter, PerformanceCounter) => time,
                    (Second, Hertz(_)) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("5_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Second, Millisecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Second, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "1_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (Second, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "1_000_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (Second, Native) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Second, PerformanceCounter) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Hertz(_)) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Second) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "1_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, Native) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Millisecond, PerformanceCounter) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Microsecond, Hertz(_)) => process
                        .integer(<BigInt as Num>::from_str_radix("5_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Microsecond, Second) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Microsecond, Millisecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Microsecond, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Microsecond, Native) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Microsecond, PerformanceCounter) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Nanosecond, Hertz(_)) => process
                        .integer(<BigInt as Num>::from_str_radix("5_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Nanosecond, Second) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Nanosecond, Millisecond) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Nanosecond, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Nanosecond, Native) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Nanosecond, PerformanceCounter) => process
                        .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                        .unwrap(),
                    (Native, Hertz(_)) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Native, Second) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (Native, Millisecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Native, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (Native, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "1_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (Native, PerformanceCounter) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Hertz(_)) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("5_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Second) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Millisecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Microsecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Nanosecond) => process
                        .integer(
                            <BigInt as Num>::from_str_radix(
                                "1_000_000_000_000_000_000_000_000",
                                10,
                            )
                            .unwrap(),
                        )
                        .unwrap(),
                    (PerformanceCounter, Native) => process
                        .integer(
                            <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10)
                                .unwrap(),
                        )
                        .unwrap(),
                };

                prop_assert_eq!(
                    native(
                        process,
                        time,
                        from_unit.to_term(process).unwrap(),
                        to_unit.to_term(process).unwrap(),
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
        .prop_filter("Term must not be a unit name", |term| {
            match term.decode().unwrap() {
                TypedTerm::Atom(atom) => match atom.name() {
                    "second" | "millisecond" | "microsecond" | "nanosecond" | "native"
                    | "perf_counter" => false,
                    _ => true,
                },
                _ => true,
            }
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
        .prop_map(move |unit| unit.to_term(&arc_process).unwrap())
        .boxed()
}

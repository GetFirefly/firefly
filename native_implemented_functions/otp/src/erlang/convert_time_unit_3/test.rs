use std::sync::Arc;

use num_bigint::BigInt;

use num_traits::Num;

use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert_eq, prop_oneof};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::time::Unit::{self, *};

use crate::erlang::convert_time_unit_3::native;
use crate::test::strategy;
use crate::test::with_process;

#[test]
fn without_integer_time_returns_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_integer(arc_process.clone()),
                unit_term(arc_process.clone()),
                unit_term(arc_process.clone()),
            )
        },
        |(arc_process, time, from_unit, to_unit)| {
            prop_assert_badarg!(
                native(&arc_process, time, from_unit, to_unit),
                format!("time ({}) must be an integer", time)
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_time_without_unit_from_unit_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                is_not_unit_term(arc_process.clone()),
                unit_term(arc_process.clone()),
            )
        },
        |(arc_process, time, from_unit, to_unit)| {
            prop_assert_is_not_time_unit!(
                native(&arc_process, time, from_unit, to_unit),
                from_unit
            );

            Ok(())
        },
    );
}

#[test]
fn with_integer_time_with_unit_from_unit_without_unit_to_unit_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_integer(arc_process.clone()),
                unit_term(arc_process.clone()),
                is_not_unit_term(arc_process.clone()),
            )
        },
        |(arc_process, time, from_unit, to_unit)| {
            prop_assert_is_not_time_unit!(native(&arc_process, time, from_unit, to_unit), to_unit);

            Ok(())
        },
    );
}

#[test]
fn with_small_integer_time_valid_units_returns_converted_value() {
    run!(
        |arc_process| (Just(arc_process.clone()), from_unit(), to_unit()),
        |(arc_process, from_unit, to_unit)| {
            // not using `proptest` for time to allow math to be hard-coded and not copy the
            // code-under-test
            let time = arc_process.integer(1_000_000_000).unwrap();

            let expected_converted = match (&from_unit, &to_unit) {
                (Hertz(_), Hertz(_)) => arc_process.integer(2_000_000_000).unwrap(),
                (Hertz(_), Second) => arc_process.integer(500_000_000).unwrap(),
                (Hertz(_), Millisecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                    .unwrap(),
                (Hertz(_), Microsecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("500_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Hertz(_), Nanosecond) => arc_process
                    .integer(
                        <BigInt as Num>::from_str_radix("500_000_000_000_000_000", 10).unwrap(),
                    )
                    .unwrap(),
                (Hertz(_), Native) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                    .unwrap(),
                (Hertz(_), PerformanceCounter) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("500_000_000_000", 10).unwrap())
                    .unwrap(),
                (Second, Second)
                | (Millisecond, Millisecond)
                | (Microsecond, Microsecond)
                | (Nanosecond, Nanosecond)
                | (Native, Native)
                | (PerformanceCounter, PerformanceCounter) => time,
                (Second, Hertz(_)) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("5_000_000_000", 10).unwrap())
                    .unwrap(),
                (Second, Millisecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Second, Microsecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Second, Nanosecond) => arc_process
                    .integer(
                        <BigInt as Num>::from_str_radix("1_000_000_000_000_000_000", 10).unwrap(),
                    )
                    .unwrap(),
                (Second, Native) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Second, PerformanceCounter) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Millisecond, Hertz(_)) => arc_process.integer(5_000_000).unwrap(),
                (Millisecond, Second) => arc_process.integer(1_000_000).unwrap(),
                (Millisecond, Microsecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Millisecond, Nanosecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Millisecond, Native) => arc_process.integer(1_000_000_000).unwrap(),
                (Millisecond, PerformanceCounter) => arc_process.integer(1_000_000_000).unwrap(),
                (Microsecond, Hertz(_)) => arc_process.integer(5_000).unwrap(),
                (Microsecond, Second) => arc_process.integer(1_000).unwrap(),
                (Microsecond, Millisecond) => arc_process.integer(1_000_000).unwrap(),
                (Microsecond, Nanosecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Microsecond, Native) => arc_process.integer(1_000_000).unwrap(),
                (Microsecond, PerformanceCounter) => arc_process.integer(1_000_000).unwrap(),
                (Nanosecond, Hertz(_)) => arc_process.integer(5).unwrap(),
                (Nanosecond, Second) => arc_process.integer(1).unwrap(),
                (Nanosecond, Millisecond) => arc_process.integer(1_000).unwrap(),
                (Nanosecond, Microsecond) => arc_process.integer(1_000_000).unwrap(),
                (Nanosecond, Native) => arc_process.integer(1_000).unwrap(),
                (Nanosecond, PerformanceCounter) => arc_process.integer(1_000).unwrap(),
                (Native, Hertz(_)) => arc_process.integer(5_000_000).unwrap(),
                (Native, Second) => arc_process.integer(1_000_000).unwrap(),
                (Native, Millisecond) => arc_process.integer(1_000_000_000).unwrap(),
                (Native, Microsecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Native, Nanosecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (Native, PerformanceCounter) => arc_process.integer(1_000_000_000).unwrap(),
                (PerformanceCounter, Hertz(_)) => arc_process.integer(5_000_000).unwrap(),
                (PerformanceCounter, Second) => arc_process.integer(1_000_000).unwrap(),
                (PerformanceCounter, Millisecond) => arc_process.integer(1_000_000_000).unwrap(),
                (PerformanceCounter, Microsecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (PerformanceCounter, Nanosecond) => arc_process
                    .integer(<BigInt as Num>::from_str_radix("1_000_000_000_000_000", 10).unwrap())
                    .unwrap(),
                (PerformanceCounter, Native) => arc_process.integer(1_000_000_000).unwrap(),
            };

            prop_assert_eq!(
                native(
                    &arc_process,
                    time,
                    from_unit.to_term(&arc_process).unwrap(),
                    to_unit.to_term(&arc_process).unwrap(),
                ),
                Ok(expected_converted)
            );

            Ok(())
        },
    );
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

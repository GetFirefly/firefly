use std::sync::Arc;

use proptest::prop_oneof;
use proptest::strategy::{BoxedStrategy, Just, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::time::Unit::{self, *};

use crate::erlang::convert_time_unit_3::result;
use crate::test::strategy;

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
                result(&arc_process, time, from_unit, to_unit),
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
                result(&arc_process, time, from_unit, to_unit),
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
            prop_assert_is_not_time_unit!(result(&arc_process, time, from_unit, to_unit), to_unit);

            Ok(())
        },
    );
}

// `with_small_integer_time_valid_units_returns_converted_value` in integration tests

// `with_big_integer_time_with_unit_from_unit_with_unit_to_unit_returns_converted_value` in
// integration tests

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

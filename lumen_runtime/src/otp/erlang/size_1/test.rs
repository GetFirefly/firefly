use proptest::prop_assert_eq;
use proptest::strategy::{Just, Strategy};
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::size_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_tuple_or_bitstring_errors_badarg() {
    TestRunner::new(Config::with_source_file(file!()))
        .run(
            &strategy::process().prop_flat_map(|arc_process| {
                (
                    Just(arc_process.clone()),
                    strategy::term(arc_process.clone())
                        .prop_filter("Term must not be a tuple or bitstring", |term| {
                            !(term.is_tuple() || term.is_bitstring())
                        }),
                )
            }),
            |(arc_process, term)| {
                prop_assert_eq!(native(&arc_process, term), Err(badarg!().into()));

                Ok(())
            },
        )
        .unwrap();
}

#[test]
fn with_tuple_returns_arity() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(0_usize..=3_usize).prop_flat_map(|size| {
                    (
                        Just(size),
                        strategy::term::tuple::intermediate(
                            strategy::term(arc_process.clone()),
                            (size..=size).into(),
                            arc_process.clone(),
                        ),
                    )
                }),
                |(size, term)| {
                    prop_assert_eq!(
                        native(&arc_process, term),
                        Ok(arc_process.integer(size).unwrap())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_bitstring_is_byte_len() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_bitstring(arc_process.clone()), |term| {
                let full_byte_len = match term.to_typed_term().unwrap() {
                    TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
                        TypedTerm::HeapBinary(heap_binary) => heap_binary.full_byte_len(),
                        TypedTerm::SubBinary(subbinary) => subbinary.full_byte_len(),
                        _ => unreachable!(),
                    },
                    _ => unreachable!(),
                };

                prop_assert_eq!(
                    native(&arc_process, term),
                    Ok(arc_process.integer(full_byte_len).unwrap())
                );

                Ok(())
            })
            .unwrap();
    });
}

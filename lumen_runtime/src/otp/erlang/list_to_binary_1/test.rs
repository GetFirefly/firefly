mod with_list;

use std::convert::TryInto;
use std::sync::Arc;

use proptest::arbitrary::any;
use proptest::strategy::{BoxedStrategy, Just, Strategy};
use proptest::test_runner::{Config, TestRunner};
use proptest::{prop_assert, prop_assert_eq, prop_oneof};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::list_to_binary_1::native;
use crate::scheduler::{with_process, with_process_arc};
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(
                    native(&arc_process, list),
                    Err(badarg!(&arc_process).into())
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        assert_eq!(
            native(process, Term::NIL),
            Ok(process.binary_from_bytes(&[]).unwrap())
        );
    });
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > list_to_binary([Bin1,1,[2,3,Bin2],4|Bin3]).
// <<1,2,3,1,2,3,4,5,4,6>>
#[test]
fn otp_doctest_returns_binary() {
    with_process(|process| {
        let bin1 = process.binary_from_bytes(&[1, 2, 3]).unwrap();
        let bin2 = process.binary_from_bytes(&[4, 5]).unwrap();
        let bin3 = process.binary_from_bytes(&[6]).unwrap();

        let iolist = process
            .improper_list_from_slice(
                &[
                    bin1,
                    process.integer(1).unwrap(),
                    process
                        .list_from_slice(&[
                            process.integer(2).unwrap(),
                            process.integer(3).unwrap(),
                            bin2,
                        ])
                        .unwrap(),
                    process.integer(4).unwrap(),
                ],
                bin3,
            )
            .unwrap();

        assert_eq!(
            native(process, iolist),
            Ok(process
                .binary_from_bytes(&[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],)
                .unwrap())
        )
    });
}

#[test]
fn with_recursive_lists_of_binaries_and_bytes_ending_in_binary_or_empty_list_returns_binary() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&top(arc_process.clone()), |list| {
                let result = native(&arc_process, list);

                prop_assert!(result.is_ok());

                Ok(())
            })
            .unwrap();
    });
}

fn byte(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    any::<u8>()
        .prop_map(move |byte| arc_process.integer(byte).unwrap())
        .boxed()
}

fn container(element: BoxedStrategy<Term>, arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        proptest::collection::vec(element, 0..=3),
        tail(arc_process.clone()),
    )
        .prop_map(move |(element_vec, tail)| {
            arc_process
                .improper_list_from_slice(&element_vec, tail)
                .unwrap()
        })
        .boxed()
}

fn leaf(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::is_binary(arc_process.clone()),
        byte(arc_process),
    ]
    .boxed()
}

fn recursive(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    leaf(arc_process.clone())
        .prop_recursive(3, 3 * 4, 3, move |element| {
            container(element, arc_process.clone())
        })
        .boxed()
}

fn tail(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![strategy::term::is_binary(arc_process), Just(Term::NIL)].boxed()
}

fn top(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    (
        proptest::collection::vec(recursive(arc_process.clone()), 1..=4),
        tail(arc_process.clone()),
    )
        .prop_map(move |(element_vec, tail)| {
            arc_process
                .improper_list_from_slice(&element_vec, tail)
                .unwrap()
        })
        .boxed()
}

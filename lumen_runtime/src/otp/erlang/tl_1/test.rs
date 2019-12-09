use proptest::prop_assert_eq;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::Term;

use crate::otp::erlang::tl_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_badarg!(
                    native(list),
                    format!("list ({}) must be a non-empty list", list)
                );

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_errors_badarg() {
    assert_badarg!(native(Term::NIL), "list ([]) must be a non-empty list");
}

#[test]
fn with_list_returns_tail() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(head, tail)| {
                    let list = arc_process.cons(head, tail).unwrap();

                    prop_assert_eq!(native(list), Ok(tail));

                    Ok(())
                },
            )
            .unwrap();
    });
}

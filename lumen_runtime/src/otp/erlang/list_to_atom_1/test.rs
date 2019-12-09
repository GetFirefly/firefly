use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::{Atom, Term};

use crate::otp::erlang::list_to_atom_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_badarg!(native(list), format!("list ({}) is not a list", list));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_atom() {
    assert_eq!(native(Term::NIL), Ok(Atom::str_to_term("")));
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |tail| {
                let list = arc_process
                    .cons(arc_process.integer('c').unwrap(), tail)
                    .unwrap();

                prop_assert_badarg!(native(list), format!("list ({}) is improper", list));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_non_empty_proper_list_returns_atom() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<String>().prop_map(|string| {
                    let codepoint_terms: Vec<Term> = string
                        .chars()
                        .map(|c| arc_process.integer(c).unwrap())
                        .collect();
                    let list = arc_process.list_from_slice(&codepoint_terms).unwrap();

                    (list, string)
                }),
                |(list, string)| {
                    let atom = Atom::str_to_term(&string);

                    prop_assert_eq!(native(list), Ok(atom));

                    Ok(())
                },
            )
            .unwrap();
    });
}

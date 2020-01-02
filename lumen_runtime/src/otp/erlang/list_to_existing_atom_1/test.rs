use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::erts::term::prelude::*;

use crate::otp::erlang::list_to_existing_atom_1::native;
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
fn with_empty_list() {
    let list = Term::NIL;

    // as `""` can only be entered into the global atom table, can't test with non-existing atom
    let existing_atom = Atom::str_to_term("");

    assert_eq!(native(list), Ok(existing_atom));
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::is_not_list(arc_process.clone()).prop_map(|tail| {
                    arc_process
                        .cons(arc_process.integer('a').unwrap(), tail)
                        .unwrap()
                }),
                |list| {
                    prop_assert_badarg!(native(list), format!("list ({}) is improper", list));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_list_without_existing_atom_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<String>().prop_map(|suffix| {
                    let string = strategy::term::non_existent_atom(&suffix);
                    let codepoint_terms: Vec<Term> = string
                        .chars()
                        .map(|c| arc_process.integer(c).unwrap())
                        .collect();

                    arc_process.list_from_slice(&codepoint_terms).unwrap()
                }),
                |list| {
                    prop_assert_badarg!(
                        native(list),
                        "tried to convert to an atom that doesn't exist"
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
// collisions due to Unicode escapes.  Could be a normalization/canonicalization issue?
#[ignore]
fn with_list_with_existing_atom_returns_atom() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &any::<String>().prop_map(|string| {
                    let codepoint_terms: Vec<Term> = string
                        .chars()
                        .map(|c| arc_process.integer(c).unwrap())
                        .collect();

                    (
                        arc_process.list_from_slice(&codepoint_terms).unwrap(),
                        Atom::str_to_term(&string),
                    )
                }),
                |(list, atom)| {
                    prop_assert_eq!(native(list), Ok(atom));

                    Ok(())
                },
            )
            .unwrap();
    });
}

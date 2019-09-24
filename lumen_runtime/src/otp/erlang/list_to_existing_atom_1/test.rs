use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::{atom_unchecked, Term};

use crate::otp::erlang::list_to_existing_atom_1::native;
use crate::scheduler::with_process_arc;
use crate::test::strategy;

#[test]
fn without_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(&strategy::term::is_not_list(arc_process.clone()), |list| {
                prop_assert_eq!(native(list), Err(badarg!().into()));

                Ok(())
            })
            .unwrap();
    });
}

#[test]
fn with_empty_list() {
    let list = Term::NIL;

    // as `""` can only be entered into the global atom table, can't test with non-existing atom
    let existing_atom = atom_unchecked("");

    assert_eq!(native(list), Ok(existing_atom));
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(native(list), Err(badarg!().into()));

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
                    prop_assert_eq!(native(list), Err(badarg!().into()));

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
                        atom_unchecked(&string),
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

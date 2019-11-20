use proptest::arbitrary::any;
use proptest::prop_assert_eq;
use proptest::strategy::Strategy;
use proptest::test_runner::{Config, TestRunner};

use liblumen_alloc::badarg;
use liblumen_alloc::erts::term::prelude::{Atom, Term};

use crate::otp::erlang::list_to_atom_1::native;
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
fn with_empty_list_returns_empty_atom() {
    with_process(|process| {
        assert_eq!(native(process, Term::NIL), Ok(Atom::str_to_term("")));
    });
}

#[test]
fn with_improper_list_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::list::improper(arc_process.clone()),
                |list| {
                    prop_assert_eq!(
                        native(&arc_process, list),
                        Err(badarg!(&arc_process).into())
                    );

                    Ok(())
                },
            )
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

                    prop_assert_eq!(native(&arc_process, list), Ok(atom));

                    Ok(())
                },
            )
            .unwrap();
    });
}

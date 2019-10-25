mod with_decimals;
mod with_scientific;

use super::*;

use std::sync::Arc;

use proptest::strategy::{BoxedStrategy, Strategy};

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::TypedTerm;

#[test]
fn without_valid_option_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::float(arc_process.clone()),
                    is_not_option(arc_process.clone())
                        .prop_map(|option| arc_process.list_from_slice(&[option]).unwrap()),
                ),
                |(float, options)| {
                    prop_assert_eq!(native(&arc_process, float, options), Err(badarg!().into()));

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_not_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Cannot be an option", |term| !is_option(term))
        .boxed()
}

fn is_option(term: &Term) -> bool {
    match term.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => atom.name() == "compact",
        TypedTerm::Tuple(tuple) => {
            (tuple.len() == 2) && {
                match tuple[0].to_typed_term().unwrap() {
                    TypedTerm::Atom(tag_atom) => match tag_atom.name() {
                        "decimals" => match tuple[1].to_typed_term().unwrap() {
                            TypedTerm::SmallInteger(small_integer) => {
                                let i: isize = small_integer.into();

                                0 <= i && i <= 253
                            }
                            _ => false,
                        },
                        "scientific" => match tuple[1].to_typed_term().unwrap() {
                            TypedTerm::SmallInteger(small_integer) => {
                                let i: isize = small_integer.into();

                                0 <= i && i <= 249
                            }
                            _ => false,
                        },
                        _ => false,
                    },
                    _ => false,
                }
            }
        }
        _ => false,
    }
}

use super::*;

use proptest::strategy::BoxedStrategy;

pub fn without_valid_option_errors_badarg(
    source_file: &'static str,
    result: fn(&Process, Term, Term) -> exception::Result<Term>,
) {
    run(
        source_file,
        |arc_process| {
            (
                Just(arc_process.clone()),
                super::strategy::term::float(arc_process.clone()),
                (
                    Just(arc_process.clone()),
                    is_not_option(arc_process.clone()),
                )
                    .prop_map(|(arc_process, option)| {
                        arc_process.list_from_slice(&[option]).unwrap()
                    }),
            )
        },
        |(arc_process, float, options)| {
            prop_assert_badarg!(
                result(&arc_process, float, options),
                "supported options are compact, {:decimal, 0..253}, or {:scientific, 0..249}"
            );

            Ok(())
        },
    );
}

fn is_not_option(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process)
        .prop_filter("Cannot be an option", |term| !is_option(term))
        .boxed()
}

fn is_option(term: &Term) -> bool {
    match term.decode().unwrap() {
        TypedTerm::Atom(atom) => atom.name() == "compact",
        TypedTerm::Tuple(tuple) => {
            (tuple.len() == 2) && {
                match tuple[0].decode().unwrap() {
                    TypedTerm::Atom(tag_atom) => match tag_atom.name() {
                        "decimals" => match tuple[1].decode().unwrap() {
                            TypedTerm::SmallInteger(small_integer) => {
                                let i: isize = small_integer.into();

                                0 <= i && i <= 253
                            }
                            _ => false,
                        },
                        "scientific" => match tuple[1].decode().unwrap() {
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

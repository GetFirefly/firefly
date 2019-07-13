use super::*;

use proptest::strategy::Strategy;

mod with_binary_subbinary;
mod with_byte;
mod with_heap_binary;

#[test]
fn without_byte_binary_or_list_element_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &is_not_byte_binary_nor_list(arc_process.clone())
                    .prop_map(|element| arc_process.cons(element, Term::NIL).unwrap()),
                |list| {
                    prop_assert_eq!(
                        erlang::list_to_binary_1(list, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_element_returns_empty_binary() {
    with_process(|process| {
        let iolist = process.cons(Term::NIL, Term::NIL).unwrap();

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(process.binary_from_bytes(&[]).unwrap())
        );
    })
}

#[test]
fn with_subbinary_with_bit_count_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::is_not_binary(arc_process.clone())
                    .prop_map(|element| arc_process.cons(element, Term::NIL).unwrap()),
                |list| {
                    prop_assert_eq!(
                        erlang::list_to_binary_1(list, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

fn is_not_byte_binary_nor_list(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_filter("Element must not be a binary or byte", move |element| {
            !(element.is_binary()
                || (element.is_integer()
                    && &arc_process.integer(0) <= element
                    && element <= &arc_process.integer(256_isize))
                || element.is_list())
        })
        .boxed()
}

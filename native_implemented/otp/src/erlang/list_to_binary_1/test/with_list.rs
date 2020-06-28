use super::*;

use proptest::strategy::{Just, Strategy};

mod with_binary_subbinary;
mod with_byte;
mod with_heap_binary;

#[test]
fn without_byte_binary_or_list_element_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                is_not_byte_binary_nor_list(arc_process.clone()),
            )
                .prop_map(|(arc_process, element)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(element, Term::NIL).unwrap(),
                        element,
                    )
                })
        },
        |(arc_process, iolist, element)| {
            prop_assert_badarg!(
                result(&arc_process, iolist),
                format!(
                    "iolist ({}) element ({}) is not a byte, binary, or nested iolist",
                    iolist, element
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_element_returns_empty_binary() {
    with_process(|process| {
        let iolist = process.cons(Term::NIL, Term::NIL).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[]).unwrap())
        );
    })
}

#[test]
fn with_subbinary_with_bit_count_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_not_binary(arc_process.clone()),
            )
        },
        |(arc_process, element)| {
            let iolist = arc_process.list_from_slice(&[element]).unwrap();

            prop_assert_badarg!(
                result(&arc_process, iolist),
                format!(
                    "iolist ({}) element ({}) is not a byte, binary, or nested iolist",
                    iolist, element
                )
            );

            Ok(())
        },
    );
}

fn is_integer_is_not_byte(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    prop_oneof![
        strategy::term::integer::negative(arc_process.clone()),
        (Just(arc_process.clone()), (256..=SmallInteger::MAX_VALUE))
            .prop_map(|(arc_process, i)| arc_process.integer(i).unwrap()),
        strategy::term::integer::big::positive(arc_process)
    ]
    .boxed()
}

fn is_not_byte_binary_nor_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_filter("Element must not be a binary or byte", move |element| {
            !(element.is_binary()
                || (element.is_integer()
                    && &arc_process.integer(0).unwrap() <= element
                    && element <= &arc_process.integer(256_isize).unwrap())
                || element.is_list())
        })
        .boxed()
}

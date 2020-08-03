use super::*;

mod with_0_bit_subbinary;
mod with_1_bit_subbinary;
mod with_2_bit_subbinary;
mod with_3_bit_subbinary;
mod with_4_bit_subbinary;
mod with_5_bit_subbinary;
mod with_6_bit_subbinary;
mod with_7_bit_subbinary;
mod with_byte;
mod with_heap_binary;

#[test]
fn without_byte_bitstring_or_list_element_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                is_not_byte_bitstring_nor_list(arc_process.clone()),
            )
                .prop_map(|(arc_process, element)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(element, Term::NIL).unwrap(),
                        element,
                    )
                })
        },
        |(arc_process, bitstring_list, element)| {
            prop_assert_badarg!(result(&arc_process, bitstring_list), format!("bitstring_list ([{}]) element ({}) is not a byte, bitstring, or nested bitstring_list", element, element));

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        let iolist = process.cons(Term::NIL, Term::NIL).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[]).unwrap())
        );
    })
}

fn element_context(bitstring_list: Term, element: Term) -> String {
    format!(
        "bitstring_list ({}) element ({}) is not a byte, bitstring, or nested bitstring_list",
        bitstring_list, element
    )
}

fn is_not_byte_bitstring_nor_list(arc_process: Arc<Process>) -> BoxedStrategy<Term> {
    strategy::term(arc_process.clone())
        .prop_filter("Element must not be a binary or byte", move |element| {
            !(element.is_bitstring()
                || (element.is_integer()
                    && &arc_process.integer(0).unwrap() <= element
                    && element <= &arc_process.integer(256).unwrap())
                || element.is_list())
        })
        .boxed()
}

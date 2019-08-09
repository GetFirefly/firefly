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
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &is_not_byte_bitstring_nor_list(arc_process.clone())
                    .prop_map(|element| arc_process.cons(element, Term::NIL).unwrap()),
                |list| {
                    prop_assert_eq!(
                        erlang::list_to_bitstring_1(list, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        let iolist = process.cons(Term::NIL, Term::NIL).unwrap();

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(process.binary_from_bytes(&[]).unwrap())
        );
    })
}

fn is_not_byte_bitstring_nor_list(arc_process: Arc<ProcessControlBlock>) -> BoxedStrategy<Term> {
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

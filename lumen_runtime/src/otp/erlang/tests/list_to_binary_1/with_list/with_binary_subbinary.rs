use super::*;

#[test]
fn without_byte_binary_or_list_element_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::sub::is_binary(arc_process.clone()),
                    is_not_byte_binary_nor_list(arc_process.clone()),
                )
                    .prop_map(|(head, tail)| arc_process.cons(head, tail).unwrap()),
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
fn with_empty_list_returns_binary_containing_subbinary_bytes() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::is_binary(arc_process.clone())
                    .prop_map(|element| (element, arc_process.cons(element, Term::NIL).unwrap())),
                |(element, list)| {
                    let subbinary: SubBinary = element.try_into().unwrap();
                    let byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();
                    let binary = arc_process.binary_from_bytes(&byte_vec).unwrap();

                    prop_assert_eq!(erlang::list_to_binary_1(list, &arc_process), Ok(binary));

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_byte_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::sub::is_binary(arc_process.clone()),
                    byte(arc_process.clone()),
                )
                    .prop_map(|(head, tail)| arc_process.cons(head, tail).unwrap()),
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
fn with_list_without_byte_tail_returns_binary() {
    with(|head, process| {
        let tail_head_byte = 254;
        let tail_head = process.integer(tail_head_byte).unwrap();

        let tail_tail = Term::NIL;

        let tail = process.cons(tail_head, tail_tail).unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(process.binary_from_bytes(&[255, 254]).unwrap())
        );
    })
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = process.binary_from_bytes(&[254, 253]).unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(process.binary_from_bytes(&[255, 254, 253]).unwrap())
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head, process| {
        let original = process
            .binary_from_bytes(&[0b0111_1111, 0b0000_0000])
            .unwrap();
        let tail = process
            .subbinary_from_original(original, 0, 1, 1, 0)
            .unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(process.binary_from_bytes(&[255, 254]).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let original = process
            .binary_from_bytes(&[0b0111_1111, 0b1000_0000])
            .unwrap();
        let head = process
            .subbinary_from_original(original, 0, 1, 1, 0)
            .unwrap();

        f(head, &process);
    })
}

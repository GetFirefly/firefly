use super::*;

#[test]
fn with_integer_without_byte_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_binary(arc_process.clone()),
                is_integer_is_not_byte(arc_process.clone()),
            )
                .prop_map(|(arc_process, head, tail)| {
                    (arc_process.clone(), arc_process.cons(head, tail), tail)
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
fn with_empty_list_returns_binary_containing_subbinary_bytes() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_binary(arc_process.clone()),
            )
                .prop_map(|(arc_process, element)| {
                    (
                        arc_process.clone(),
                        element,
                        arc_process.cons(element, Term::Nil),
                    )
                })
        },
        |(arc_process, element, list)| {
            let subbinary: Boxed<SubBinary> = element.try_into().unwrap();
            let byte_vec: Vec<u8> = subbinary.full_byte_iter().collect();
            let binary = arc_process.binary_from_bytes(&byte_vec);

            prop_assert_eq!(result(&arc_process, list), Ok(binary));

            Ok(())
        },
    );
}

#[test]
fn with_byte_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::sub::is_binary(arc_process.clone()),
                byte(arc_process.clone()),
            )
        },
        |(arc_process, head, tail)| {
            let iolist = arc_process.cons(head, tail);

            prop_assert_badarg!(
                result(&arc_process, iolist),
                format!("iolist ({}) tail ({}) cannot be a byte", iolist, tail)
            );

            Ok(())
        },
    );
}

#[test]
fn with_list_without_byte_tail_returns_binary() {
    with(|head, process| {
        let tail_head_byte = 254;
        let tail_head = process.integer(tail_head_byte).unwrap();

        let tail_tail = Term::Nil;

        let tail = process.cons(tail_head, tail_tail);

        let iolist = process.cons(head, tail);

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[255, 254]))
        );
    })
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = process.binary_from_bytes(&[254, 253]);

        let iolist = process.cons(head, tail);

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[255, 254, 253]))
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head, process| {
        let original = process.binary_from_bytes(&[0b0111_1111, 0b0000_0000]);
        let tail = process.subbinary_from_original(original, 0, 1, 1, 0);

        let iolist = process.cons(head, tail);

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[255, 254]))
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let original = process.binary_from_bytes(&[0b0111_1111, 0b1000_0000]);
        let head = process.subbinary_from_original(original, 0, 1, 1, 0);

        f(head, &process);
    })
}

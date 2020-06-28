use super::*;

#[test]
fn with_integer_without_byte_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::heap(arc_process.clone()),
                is_integer_is_not_byte(arc_process.clone()),
            )
                .prop_map(|(arc_process, head, tail)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(head, tail).unwrap(),
                        tail,
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
fn with_empty_list_returns_binary() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::binary::heap(arc_process.clone()),
            )
                .prop_map(|(arc_process, binary)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(binary, Term::NIL).unwrap(),
                        binary,
                    )
                })
        },
        |(arc_process, list, binary)| {
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
                strategy::term::binary::heap(arc_process.clone()),
                byte(arc_process.clone()),
            )
                .prop_map(|(arc_process, head, tail)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(head, tail).unwrap(),
                        tail,
                    )
                })
        },
        |(arc_process, iolist, tail)| {
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
        let tail_head_byte = 3;
        let tail_head = process.integer(tail_head_byte).unwrap();
        let tail_tail = Term::NIL;
        let tail = process.cons(tail_head, tail_tail).unwrap();
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[0, 1, tail_head_byte]).unwrap())
        );
    })
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = process.binary_from_bytes(&[2, 3]).unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[0, 1, 2, 3]).unwrap())
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head, process| {
        let original = process
            .binary_from_bytes(&[0b0111_1111, 0b1000_0000])
            .unwrap();
        let tail = process
            .subbinary_from_original(original, 0, 1, 1, 0)
            .unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[0, 1, 255]).unwrap())
        );
    })
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let head = process.binary_from_bytes(&[0, 1]).unwrap();

        f(head, &process);
    })
}

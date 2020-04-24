use super::*;

#[test]
fn without_byte_bitstring_or_list_element_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                byte(arc_process.clone()),
                is_not_byte_bitstring_nor_list(arc_process.clone()),
            )
                .prop_map(|(arc_process, head, tail)| {
                    (
                        arc_process.clone(),
                        arc_process.cons(head, tail).unwrap(),
                        tail,
                    )
                })
        },
        |(arc_process, bitstring_list, element)| {
            prop_assert_badarg!(
                result(&arc_process, bitstring_list),
                element_context(bitstring_list, element)
            );

            Ok(())
        },
    );
}

#[test]
fn with_empty_list_returns_1_byte_binary() {
    with(|head_byte, head, process| {
        let tail = Term::NIL;
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[head_byte]).unwrap())
        );
    })
}

#[test]
fn with_byte_tail_errors_badarg() {
    with_tail_errors_badarg(|process| process.integer(2).unwrap());
}

#[test]
fn with_proper_list_returns_binary() {
    with(|_, head, process| {
        let tail_head_byte = 1;
        let tail_head = process.integer(tail_head_byte).unwrap();
        let tail_tail = Term::NIL;
        let tail = process.cons(tail_head, tail_tail).unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[0, 1]).unwrap())
        );
    });
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head_byte, head, process| {
        let tail = process.binary_from_bytes(&[1, 2]).unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[head_byte, 1, 2]).unwrap())
        );
    })
}

#[test]
fn with_subbinary_with_bit_count_0_returns_binary() {
    with(|head_byte, head, process| {
        let original = process.binary_from_bytes(&[1]).unwrap();
        let tail = process
            .subbinary_from_original(original, 0, 0, 1, 0)
            .unwrap();

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(process.binary_from_bytes(&[head_byte, 1]).unwrap())
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_1_returns_subbinary() {
    with(|head_byte, head, process| {
        let tail = bitstring!(2, 0b1 :: 1, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(head_byte, 2, 0b1 :: 1, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_2_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 3 :: 2, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 3 :: 2, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_3_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 0b101 :: 3, &process);

        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 0b101 :: 3, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_4_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 0b0101 :: 4, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 0b0101 :: 4, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_5_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 0b10101 :: 5, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 0b10101 :: 5, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_6_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 0b010101 :: 6, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 0b010101 :: 6, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_7_returns_subbinary() {
    with(|_head_byte, head, process| {
        let tail = bitstring!(1, 0b1010101 :: 7, &process);
        let iolist = process.cons(head, tail).unwrap();

        assert_eq!(
            result(process, iolist),
            Ok(bitstring!(0, 1, 0b1010101 :: 7, &process))
        )
    });
}

fn with_tail_errors_badarg<T>(tail: T)
where
    T: FnOnce(&Process) -> Term,
{
    with(|_, head, process| {
        let tail = tail(&process);
        let bitstring_list = process.cons(head, tail).unwrap();

        assert_badarg!(
            result(process, bitstring_list),
            format!(
                "bitstring_list ({}) tail ({}) cannot be a byte",
                bitstring_list, tail
            )
        );
    });
}

fn with<F>(f: F)
where
    F: FnOnce(u8, Term, &Process) -> (),
{
    with_process(|process| {
        let head_byte: u8 = 0;
        let head = process.integer(head_byte).unwrap();

        f(head_byte, head, &process);
    })
}

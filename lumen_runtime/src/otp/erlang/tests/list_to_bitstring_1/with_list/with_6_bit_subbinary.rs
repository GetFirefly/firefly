use super::*;

#[test]
fn without_byte_bitstring_or_list_element_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::binary::sub::with_bit_count(5, arc_process.clone()),
                    is_not_byte_bitstring_nor_list(arc_process.clone()),
                )
                    .prop_map(|(head, tail)| Term::cons(head, tail, &arc_process)),
                |list| {
                    prop_assert_eq!(
                        erlang::list_to_bitstring_1(list, &arc_process),
                        Err(badarg!())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_bitstring() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &strategy::term::binary::sub::with_bit_count(5, arc_process.clone())
                    .prop_map(|head| (Term::cons(head, Term::EMPTY_LIST, &arc_process), head)),
                |(list, bitstring)| {
                    prop_assert_eq!(
                        erlang::list_to_bitstring_1(list, &arc_process),
                        Ok(bitstring)
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_empty_list_returns_binary() {
    with(|head, process| {
        let tail = Term::EMPTY_LIST;
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(erlang::list_to_bitstring_1(iolist, &process), Ok(head));
    })
}

#[test]
fn with_improper_list_returns_binary() {
    with_tail_errors_badarg(|process| {
        let tail_head = 254.into_process(&process);
        let tail_tail = 253.into_process(&process);

        Term::cons(tail_head, tail_tail, &process)
    });
}

#[test]
fn with_proper_list_returns_binary() {
    with(|head, process| {
        let tail_head_byte = 254;
        let tail_head = tail_head_byte.into_process(&process);
        let tail_tail = Term::EMPTY_LIST;
        let tail = Term::cons(tail_head, tail_tail, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 87, 62 :: 6, &process))
        );
    });
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = Term::slice_to_binary(&[254, 253], &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 87, 251, 61 :: 6, &process))
        );
    })
}

#[test]
fn with_subbinary_with_bit_count_0_returns_binary() {
    with(|head, process| {
        let original = Term::slice_to_binary(&[2], &process);
        let tail = Term::subbinary(original, 0, 0, 1, 0, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!( 1, 84, 2 :: 6, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_1_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(2, 0b1 :: 1, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 5 :: 7, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_2_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b11 :: 2, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(Term::slice_to_binary(&[1, 84, 11], &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_3_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b101 :: 3, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 10, 1 :: 1, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_4_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b0101 :: 4, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 9, 1 :: 2, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_5_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b10101 :: 5, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 10, 5 :: 3, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_6_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b010101 :: 6, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 9, 5 :: 4, &process))
        );
    });
}

#[test]
fn with_subbinary_with_bit_count_7_returns_subbinary() {
    with(|head, process| {
        let tail = bitstring!(0b0000_0010, 0b1010101 :: 7, &process);
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 84, 10, 21 :: 5, &process))
        )
    });
}

fn with_tail_errors_badarg<T>(tail: T)
where
    T: FnOnce(&Process) -> Term,
{
    with(|head, process| {
        let iolist = Term::cons(head, tail(&process), &process);

        assert_badarg!(erlang::list_to_bitstring_1(iolist, &process));
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let head = bitstring!(1, 0b010101 :: 6, &process);

        f(head, &process);
    })
}

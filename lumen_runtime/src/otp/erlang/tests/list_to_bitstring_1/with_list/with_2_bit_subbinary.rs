use super::*;

#[test]
fn with_atom_errors_badarg() {
    with_tail_errors_badarg(|_| Term::str_to_atom("", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    with_tail_errors_badarg(|process| Term::next_local_reference(process));
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
    })
}

#[test]
fn with_proper_list_returns_binary() {
    with(|head, process| {
        let tail_head = 254.into_process(&process);
        let tail_tail = Term::EMPTY_LIST;
        let tail = Term::cons(tail_head, tail_tail, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 255, 2 :: 2, &process))
        );
    })
}

#[test]
fn with_byte_errors_badarg() {
    with_tail_errors_badarg(|process| 254.into_process(&process));
}

#[test]
fn with_small_integer_with_byte_overflow_errors_badarg() {
    with_tail_errors_badarg(|process| 256.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    with_tail_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    with_tail_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    with_tail_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    with_tail_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    with_tail_errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    with_tail_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_returns_binary() {
    with(|head, process| {
        let tail = Term::slice_to_binary(&[254, 253], &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(bitstring!(1, 255, 191, 1 :: 2, &process))
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
            Ok(bitstring!(1, 192, 2 :: 2, &process))
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
            Ok(bitstring!(1, 192, 5 :: 3, &process))
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
            Ok(bitstring!(1, 192, 11 :: 4, &process))
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
            Ok(bitstring!(1, 192, 21 :: 5, &process))
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
            Ok(bitstring!(1, 192, 37 :: 6, &process))
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
            Ok(bitstring!(1, 192, 85 :: 7, &process))
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
            Ok(Term::slice_to_binary(&[1, 192, 149], &process))
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
            Ok(bitstring!(1, 192, 170, 1 :: 1, &process)),
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
        let head = bitstring!(1, 0b11 :: 2, &process);

        f(head, &process);
    })
}

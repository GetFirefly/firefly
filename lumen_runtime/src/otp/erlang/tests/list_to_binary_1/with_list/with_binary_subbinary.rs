use super::*;

#[test]
fn with_atom_errors_badarg() {
    with_tail_errors_badarg(|_| Term::str_to_atom("", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    with_tail_errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_returns_binary() {
    with(|head, process| {
        let tail = Term::EMPTY_LIST;
        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[255], &process))
        );
    })
}

#[test]
fn with_list_with_byte_tail_errors_badarg() {
    with_tail_errors_badarg(|process| {
        let tail_head_byte = 254;
        let tail_head = tail_head_byte.into_process(&process);

        let tail_tail_byte = 253;
        let tail_tail = tail_tail_byte.into_process(&process);

        Term::cons(tail_head, tail_tail, &process)
    });
}

#[test]
fn with_list_without_byte_tail_returns_binary() {
    with(|head, process| {
        let tail_head_byte = 254;
        let tail_head = tail_head_byte.into_process(&process);

        let tail_tail = Term::EMPTY_LIST;

        let tail = Term::cons(tail_head, tail_tail, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[255, 254], &process))
        );
    })
}

#[test]
fn with_byte_returns_errors_badarg() {
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
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[255, 254, 253], &process))
        );
    })
}

#[test]
fn with_subbinary_without_bitcount_returns_binary() {
    with(|head, process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b0000_0000], &process);
        let tail = Term::subbinary(original, 0, 1, 1, 0, &process);

        let iolist = Term::cons(head, tail, &process);

        assert_eq!(
            erlang::list_to_binary_1(iolist, &process),
            Ok(Term::slice_to_binary(&[255, 254], &process))
        );
    })
}

#[test]
fn with_subbinary_with_bitcount_errors_badarg() {
    with_tail_errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1100_0000], &process);
        Term::subbinary(original, 0, 1, 1, 1, &process)
    })
}

fn with_tail_errors_badarg<T>(tail: T)
where
    T: FnOnce(&Process) -> Term,
{
    with(|head, process| {
        let iolist = Term::cons(head, tail(&process), &process);

        assert_badarg!(erlang::list_to_binary_1(iolist, &process));
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, &Process) -> (),
{
    with_process(|process| {
        let original = Term::slice_to_binary(&[0b0111_1111, 0b1000_0000], &process);
        let head = Term::subbinary(original, 0, 1, 1, 0, &process);

        f(head, &process);
    })
}

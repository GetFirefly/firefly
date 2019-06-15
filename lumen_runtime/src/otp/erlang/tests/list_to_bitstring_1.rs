use super::*;

mod with_list;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("iolist", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_returns_empty_binary() {
    with_process(|process| {
        assert_eq!(
            erlang::list_to_bitstring_1(Term::EMPTY_LIST, &process),
            Ok(Term::slice_to_binary(&[], &process))
        );
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

// > Bin1 = <<1,2,3>>.
// <<1,2,3>>
// > Bin2 = <<4,5>>.
// <<4,5>>
// > Bin3 = <<6>>.
// <<6>>
// > list_to_binary([Bin1,1,[2,3,Bin2],4|Bin3]).
// <<1,2,3,1,2,3,4,5,4,6>>
#[test]
fn otp_doctest_returns_binary() {
    with_process(|process| {
        let bin1 = Term::slice_to_binary(&[1, 2, 3], &process);
        let bin2 = Term::slice_to_binary(&[4, 5], &process);
        let bin3 = Term::slice_to_binary(&[6], &process);

        let iolist = Term::slice_to_improper_list(
            &[
                bin1,
                1.into_process(&process),
                Term::slice_to_list(
                    &[2.into_process(&process), 3.into_process(&process), bin2],
                    &process,
                ),
                4.into_process(&process),
            ],
            bin3,
            &process,
        );

        assert_eq!(
            erlang::list_to_bitstring_1(iolist, &process),
            Ok(Term::slice_to_binary(
                &[1, 2, 3, 1, 2, 3, 4, 5, 4, 6],
                &process
            ))
        )
    });
}

#[test]
fn with_subbinary_errors_badarg() {
    errors_badarg(|process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(binary_term, 0, 7, 2, 1, &process)
    });
}

fn errors_badarg<I>(iolist: I)
where
    I: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::list_to_bitstring_1(iolist(&process), &process))
}

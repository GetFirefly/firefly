use super::*;

mod with_small_integer_position;

#[test]
fn with_atom_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_position_errors_badarg() {
    with_position_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_position_errors_badarg() {
    with_position_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_big_integer_position_errors_badarg() {
    with_position_errors_badarg(|process| (crate::integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_position_errors_badarg() {
    with_position_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_position_errors_badarg() {
    with_position_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_position_errors_badarg() {
    with_position_errors_badarg(|process| {
        Term::slice_to_tuple(
            &[0.into_process(&process), 1.into_process(&process)],
            &process,
        )
    });
}

fn with_position_errors_badarg<P>(position: P)
where
    P: FnOnce(&Process) -> Term,
{
    errors_badarg(|process| {
        let binary = bitstring!(1 :: 1, &process);

        erlang::split_binary_2(binary, position(&process), &process)
    })
}

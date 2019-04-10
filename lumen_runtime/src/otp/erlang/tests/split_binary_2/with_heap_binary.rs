use super::*;

mod with_small_integer_position;

#[test]
fn with_atom_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_position_errors_badarg() {
    with_position_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_position_errors_badarg() {
    with_position_errors_badarg(|mut process| list_term(&mut process));
}

#[test]
fn with_big_integer_position_errors_badarg() {
    with_position_errors_badarg(|mut process| {
        (crate::integer::small::MAX + 1).into_process(&mut process)
    });
}

#[test]
fn with_float_position_errors_badarg() {
    with_position_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_position_errors_badarg() {
    with_position_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_position_errors_badarg() {
    with_position_errors_badarg(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_position_errors_badarg() {
    with_position_errors_badarg(|mut process| {
        Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process,
        )
    });
}

fn with_position_errors_badarg<P>(position: P)
where
    P: FnOnce(&mut Process) -> Term,
{
    errors_badarg(|mut process| {
        let binary = Term::slice_to_binary(&[1], &mut process);

        erlang::split_binary_2(binary, position(&mut process), &mut process)
    })
}

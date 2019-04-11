use super::*;

mod with_false_left;
mod with_true_left;

#[test]
fn with_atom_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::str_to_atom("left", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_left_errors_badarg() {
    with_left_errors_badarg(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_left_errors_badarg() {
    with_left_errors_badarg(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_float_errors_badarg() {
    with_left_errors_badarg(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_left_errors_badarg() {
    with_left_errors_badarg(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_left_errors_badarg() {
    with_left_errors_badarg(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_left_errors_badarg() {
    with_left_errors_badarg(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_left_errors_badarg() {
    with_left_errors_badarg(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_left_errors_badarg() {
    with_left_errors_badarg(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_left_errors_badarg() {
    with_left_errors_badarg(|mut process| bitstring!(1 ::1, &mut process));
}

fn with_left_errors_badarg<L>(left: L)
where
    L: FnOnce(&mut Process) -> Term,
{
    super::errors_badarg(|mut process| {
        let left = left(&mut process);
        let right = 0.into_process(&mut process);

        erlang::xor_2(left, right)
    });
}

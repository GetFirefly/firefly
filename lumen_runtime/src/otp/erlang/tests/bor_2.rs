use super::*;

mod with_big_integer_left;
mod with_small_integer_left;

#[test]
fn with_atom_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::str_to_atom("left", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_left_errors_badarith() {
    with_left_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_left_errors_badarith() {
    with_left_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_float_errors_badarith() {
    with_left_errors_badarith(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_left_errors_badarith() {
    with_left_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_left_errors_badarith() {
    with_left_errors_badarith(|mut process| Term::external_pid(1, 2, 3, &mut process).unwrap());
}

#[test]
fn with_tuple_left_errors_badarith() {
    with_left_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_left_errors_badarith() {
    with_left_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_left_errors_badarith() {
    with_left_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_left_errors_badarith() {
    with_left_errors_badarith(|mut process| bitstring!(1 ::1, &mut process));
}

fn with_left_errors_badarith<L>(left: L)
where
    L: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let left = left(&mut process);
        let right = 0.into_process(&mut process);

        erlang::bor_2(left, right, &mut process)
    });
}

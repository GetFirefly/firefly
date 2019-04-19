use super::*;

#[test]
fn with_atom_right_returns_false() {
    with_right_returns_false(|mut _process| Term::str_to_atom("right", DoNotCare).unwrap());
}

#[test]
fn with_false_right_returns_false() {
    with_right_returns_false(|_| false.into());
}

#[test]
fn with_true_right_returns_false() {
    with_right_returns_false(|_| true.into());
}

#[test]
fn with_local_reference_right_returns_false() {
    with_right_returns_false(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_right_returns_false() {
    with_right_returns_false(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_right_returns_false() {
    with_right_returns_false(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_right_returns_false() {
    with_right_returns_false(|process| 1.into_process(&process))
}

#[test]
fn with_big_integer_right_returns_false() {
    with_right_returns_false(|process| (crate::integer::small::MAX + 1).into_process(&process))
}

#[test]
fn with_float_right_returns_false() {
    with_right_returns_false(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_right_returns_false() {
    with_right_returns_false(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_right_returns_false() {
    with_right_returns_false(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_right_returns_false() {
    with_right_returns_false(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_right_returns_false() {
    with_right_returns_false(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_right_returns_false() {
    with_right_returns_false(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_right_returns_false() {
    with_right_returns_false(|process| bitstring!(1 :: 1, &process));
}

fn with_right_returns_false<R>(right: R)
where
    R: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let left = false.into();
        let right = right(&process);

        assert_eq!(erlang::andalso_2(left, right), Ok(left));
    });
}

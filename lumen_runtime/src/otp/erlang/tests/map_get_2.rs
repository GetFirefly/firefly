use super::*;

use crate::process::IntoProcess;

mod with_map;

#[test]
fn with_atom_errors_badmap() {
    with_key_and_map_errors_badmap(
        |_| Term::str_to_atom("key", DoNotCare).unwrap(),
        |_| Term::str_to_atom("map", DoNotCare).unwrap(),
    );
}

#[test]
fn with_local_reference_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| Term::next_local_reference(process),
        |process| Term::next_local_reference(process),
    );
}

#[test]
fn with_empty_list_errors_badmap() {
    with_key_and_map_errors_badmap(|_| Term::EMPTY_LIST, |_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
        |process| Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process),
    );
}

#[test]
fn with_small_integer_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| 1.into_process(&process),
        |process| 0.into_process(&process),
    );
}

#[test]
fn with_big_integer_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| (integer::small::MAX + 1).into_process(&process),
        |process| (integer::small::MAX + 2).into_process(&process),
    );
}

#[test]
fn with_float_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| 2.0.into_process(&process),
        |process| 1.0.into_process(&process),
    );
}

#[test]
fn with_local_pid_errors_badmap() {
    with_key_and_map_errors_badmap(
        |_| Term::local_pid(2, 3).unwrap(),
        |_| Term::local_pid(0, 1).unwrap(),
    );
}

#[test]
fn with_external_pid_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| Term::external_pid(4, 5, 6, &process).unwrap(),
        |process| Term::external_pid(1, 2, 3, &process).unwrap(),
    );
}

#[test]
fn with_tuple_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| Term::slice_to_tuple(&[1.into_process(&process)], &process),
        |process| Term::slice_to_tuple(&[0.into_process(&process)], &process),
    );
}

#[test]
fn with_heap_binary_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| Term::slice_to_binary(&[1], &process),
        |process| Term::slice_to_binary(&[0], &process),
    );
}

#[test]
fn with_subbinary_errors_badmap() {
    with_key_and_map_errors_badmap(
        |process| bitstring!(2 :: 2, &process),
        |process| bitstring!(1 :: 1, &process),
    );
}

fn with_key_and_map_errors_badmap<K, M>(key: K, map: M)
where
    K: FnOnce(&Process) -> Term,
    M: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let key = key(&process);
        let map = map(&process);

        assert_badmap!(erlang::map_get_2(key, map, &process), map, &process);
    })
}

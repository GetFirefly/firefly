use super::*;

use crate::process::IntoProcess;

#[test]
fn with_atom_errors_badmap() {
    errors_badmap(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badmap() {
    errors_badmap(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list_errors_badmap() {
    errors_badmap(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badmap() {
    errors_badmap(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_badmap() {
    errors_badmap(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badmap() {
    errors_badmap(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badmap() {
    errors_badmap(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badmap() {
    errors_badmap(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badmap() {
    errors_badmap(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badmap() {
    errors_badmap(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_without_elements_is_zero() {
    with_process(|process| {
        let map = Term::slice_to_map(&[], &process);

        assert_eq!(
            erlang::map_size_1(map, &process),
            Ok(0.into_process(&process))
        );
    });
}

#[test]
fn with_map_with_elements_is_element_count() {
    with_process(|process| {
        let map = Term::slice_to_map(
            &[
                (
                    Term::str_to_atom("one", DoNotCare).unwrap(),
                    1.into_process(&process),
                ),
                (
                    Term::str_to_atom("two", DoNotCare).unwrap(),
                    2.into_process(&process),
                ),
            ],
            &process,
        );

        assert_eq!(
            erlang::map_size_1(map, &process),
            Ok(2.into_process(&process))
        );
    });
}
#[test]
fn with_heap_binary_errors_badmap() {
    errors_badmap(|process| Term::slice_to_binary(&[0, 1, 2], &process));
}

#[test]
fn with_subbinary_errors_badmap() {
    errors_badmap(|process| bitstring!(1 :: 1, &process));
}

fn errors_badmap<M>(map: M)
where
    M: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let map = map(&process);

        assert_badmap!(erlang::map_size_1(map, &process), map, &process);
    })
}

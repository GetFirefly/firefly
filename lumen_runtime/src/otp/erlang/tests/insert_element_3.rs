use super::*;

#[test]
fn with_atom_errors_badarg() {
    with_tuple_errors_badarg(|_| Term::str_to_atom("atom", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    with_tuple_errors_badarg(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_errors_badarg() {
    with_tuple_errors_badarg(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_errors_badarg() {
    with_tuple_errors_badarg(|process| list_term(&process));
}

#[test]
fn with_small_integer_errors_badarg() {
    with_tuple_errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    with_tuple_errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    with_tuple_errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    with_tuple_errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    with_tuple_errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_without_small_integer_index_errors_badarg() {
    with_process(|process| {
        let tuple = Term::slice_to_tuple(
            &[1.into_process(&process), 3.into_process(&process)],
            &process,
        );
        let index = 2usize;
        let invalid_index_term = Term::arity(index);

        assert_ne!(invalid_index_term.tag(), SmallInteger);
        assert_badarg!(erlang::insert_element_3(
            tuple,
            invalid_index_term,
            0.into_process(&process),
            &process
        ));

        let valid_index_term: Term = index.into_process(&process);

        assert_eq!(valid_index_term.tag(), SmallInteger);
        assert_eq!(
            erlang::insert_element_3(valid_index_term, tuple, 2.into_process(&process), &process),
            Ok(Term::slice_to_tuple(
                &[
                    1.into_process(&process),
                    2.into_process(&process),
                    3.into_process(&process)
                ],
                &process
            ))
        );
    });
}

#[test]
fn with_tuple_without_index_in_range_errors_badarg() {
    with_process(|process| {
        let tuple = Term::slice_to_tuple(&[], &process);

        assert_badarg!(erlang::insert_element_3(
            tuple,
            2.into_process(&process),
            0.into_process(&process),
            &process
        ));
    });
}

#[test]
fn with_tuple_with_index_in_range_returns_tuple_with_new_element_at_index() {
    with_process(|process| {
        let tuple = Term::slice_to_tuple(
            &[1.into_process(&process), 3.into_process(&process)],
            &process,
        );

        assert_eq!(
            erlang::insert_element_3(
                2.into_process(&process),
                tuple,
                2.into_process(&process),
                &process
            ),
            Ok(Term::slice_to_tuple(
                &[
                    1.into_process(&process),
                    2.into_process(&process),
                    3.into_process(&process)
                ],
                &process
            ))
        );
    });
}

#[test]
fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
    with_process(|process| {
        let tuple = Term::slice_to_tuple(&[1.into_process(&process)], &process);

        assert_eq!(
            erlang::insert_element_3(
                2.into_process(&process),
                tuple,
                3.into_process(&process),
                &process
            ),
            Ok(Term::slice_to_tuple(
                &[1.into_process(&process), 3.into_process(&process)],
                &process
            ))
        );
    });
}

#[test]
fn with_map_errors_badarg() {
    with_tuple_errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badarg() {
    with_tuple_errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badarg() {
    with_tuple_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_tuple_errors_badarg<T>(tuple: T)
where
    T: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| {
        erlang::insert_element_3(
            0.into_process(&process),
            tuple(&process),
            Term::str_to_atom("element", DoNotCare).unwrap(),
            &process,
        )
    });
}

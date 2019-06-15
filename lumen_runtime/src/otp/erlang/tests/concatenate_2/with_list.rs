use super::*;

#[test]
fn with_atom_return_improper_list_with_atom_as_tail() {
    with(|element, list, process| {
        let term = Term::str_to_atom("term", DoNotCare).unwrap();

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_local_reference_returns_improper_list_with_local_reference_as_tail() {
    with(|element, list, process| {
        let term = Term::next_local_reference(process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_improper_list_return_improper_list_with_improper_list_as_tail() {
    with(|element, list, process| {
        let term = Term::cons(1.into_process(&process), 2.into_process(&process), &process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_small_integer_returns_improper_list_with_small_integer_as_tail() {
    with(|element, list, process| {
        let term = 1.into_process(&process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_big_integer_returns_improper_list_with_big_integer_as_tail() {
    with(|element, list, process| {
        let term = (integer::small::MAX + 1).into_process(&process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_float_returns_improper_list_with_float_as_tail() {
    with(|element, list, process| {
        let term = 1.0.into_process(&process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_local_pid_returns_improper_list_with_local_pid_as_tail() {
    with(|element, list, process| {
        let term = Term::local_pid(1, 2).unwrap();

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_external_pid_returns_improper_list_with_external_pid_as_tail() {
    with(|element, list, process| {
        let term = Term::external_pid(4, 5, 6, &process).unwrap();

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_tuple_returns_improper_list_with_tuple_as_tail() {
    with(|element, list, process| {
        let term = Term::slice_to_tuple(&[], &process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_map_returns_improper_list_with_map_as_tail() {
    with(|element, list, process| {
        let term = Term::slice_to_map(&[], &process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_heap_binary_returns_improper_list_with_heap_binary_as_tail() {
    with(|element, list, process| {
        let term = Term::slice_to_binary(&[], &process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

#[test]
fn with_subbinary_returns_improper_list_with_subbinary_as_tail() {
    with(|element, list, process| {
        let binary_term =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        let term = Term::subbinary(binary_term, 0, 7, 2, 0, &process);

        assert_eq!(
            erlang::concatenate_2(list, term, &process),
            Ok(Term::cons(element, term, &process))
        );
    });
}

fn with<F>(f: F)
where
    F: FnOnce(Term, Term, &Process) -> (),
{
    with_process(|process| {
        let element = 0.into_process(&process);
        let list = Term::cons(element, Term::EMPTY_LIST, &process);

        f(element, list, &process)
    })
}

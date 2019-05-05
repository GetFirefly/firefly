use super::*;

// The behavior here is weird to @KronicDeth and @bitwalker, but consistent with BEAM.
// See https://bugs.erlang.org/browse/ERL-898.

#[test]
fn with_atom_returns_atom() {
    returns_term(|_| Term::str_to_atom("term", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_returns_local_reference() {
    returns_term(|process| Term::next_local_reference(process));
}

#[test]
fn with_improper_list_returns_improper_list() {
    returns_term(|process| {
        Term::cons(2.into_process(&process), 3.into_process(&process), &process)
    });
}

#[test]
fn with_small_integer_returns_small_integer() {
    returns_term(|process| 1.into_process(&process));
}

#[test]
fn with_big_integer_returns_big_integer() {
    returns_term(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_returns_float() {
    returns_term(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_returns_local_pid() {
    returns_term(|_| Term::local_pid(1, 2).unwrap());
}

#[test]
fn with_external_pid_returns_external_pid() {
    returns_term(|process| Term::external_pid(4, 5, 6, &process).unwrap());
}

#[test]
fn with_tuple_returns_tuple() {
    returns_term(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_returns_map_is() {
    returns_term(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_returns_heap_binary() {
    returns_term(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_returns_subbinary() {
    returns_term(|process| bitstring!(1 :: 1, &process));
}

fn returns_term<T>(term: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let term = term(&process);

        assert_eq!(
            erlang::concatenate_2(Term::EMPTY_LIST, term, &process),
            Ok(term)
        );
    });
}

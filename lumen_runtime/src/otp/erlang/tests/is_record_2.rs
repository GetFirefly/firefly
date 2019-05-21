use super::*;

mod with_tuple;

#[test]
fn with_atom_is_false() {
    is_record_returns_false(|_| Term::str_to_atom("term", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_is_false() {
    is_record_returns_false(|process| Term::next_local_reference(&process));
}

#[test]
fn with_empty_list_is_false() {
    is_record_returns_false(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_is_false() {
    is_record_returns_false(|process| list_term(&process));
}

#[test]
fn with_small_integer_is_false() {
    is_record_returns_false(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_is_false() {
    is_record_returns_false(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_is_false() {
    is_record_returns_false(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_is_true() {
    is_record_returns_false(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_is_true() {
    is_record_returns_false(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_map_is_false() {
    is_record_returns_false(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_is_false() {
    is_record_returns_false(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_is_false() {
    is_record_returns_false(|process| bitstring!(1 :: 1, process));
}

fn is_record_returns_false<T>(term: T)
where
    T: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

        assert_eq!(
            erlang::is_record_2(term(process), record_tag),
            Ok(false.into())
        )
    });
}

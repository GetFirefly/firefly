use super::*;

mod with_tuple;

#[test]
fn with_atom_is_false() {
    let term = Term::str_to_atom("term", DoNotCare).unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_local_reference_is_false() {
    let process = process::local::new();
    let term = Term::next_local_reference(&process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_empty_list_is_false() {
    let term = Term::EMPTY_LIST;
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_list_is_false() {
    let process = process::local::new();
    let term = list_term(&process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_small_integer_is_false() {
    let process = process::local::new();
    let term = 0.into_process(&process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_big_integer_is_false() {
    let process = process::local::new();
    let term = (integer::small::MAX + 1).into_process(&process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_float_is_false() {
    let process = process::local::new();
    let term = 1.0.into_process(&process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_local_pid_is_true() {
    let term = Term::local_pid(0, 0).unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_external_pid_is_true() {
    let process = process::local::new();
    let term = Term::external_pid(1, 0, 0, &process).unwrap();
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_map_is_false() {
    let process = process::local::new();
    let term = Term::slice_to_map(&[], &process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_heap_binary_is_false() {
    let process = process::local::new();
    let term = Term::slice_to_binary(&[], &process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

#[test]
fn with_subbinary_is_false() {
    let process = process::local::new();
    let original = Term::slice_to_binary(&[129, 0b0000_0000], &process);
    let term = Term::subbinary(original, 0, 1, 1, 0, &process);
    let record_tag = Term::str_to_atom("record_tag", DoNotCare).unwrap();

    assert_eq!(erlang::is_record_2(term, record_tag), Ok(false.into()));
}

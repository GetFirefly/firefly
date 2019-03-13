use super::*;

use num_traits::Num;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(erlang::list_to_pid(atom_term, &mut process), process);
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();

    assert_bad_argument!(erlang::list_to_pid(Term::EMPTY_LIST, &mut process), process);
}

#[test]
fn with_list_encoding_local_pid() {
    let mut process: Process = Default::default();

    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<0", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<0.", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<0.1", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<0.1.", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<0.1.2", &mut process), &mut process),
        process
    );

    assert_eq_in_process!(
        erlang::list_to_pid(
            Term::str_to_char_list("<0.1.2>", &mut process),
            &mut process
        ),
        Term::local_pid(1, 2),
        process
    );

    assert_bad_argument!(
        erlang::list_to_pid(
            Term::str_to_char_list("<0.1.2>?", &mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_list_encoding_external_pid() {
    let mut process: Process = Default::default();

    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<1", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<1.", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<1.2", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<1.2.", &mut process), &mut process),
        process
    );
    assert_bad_argument!(
        erlang::list_to_pid(Term::str_to_char_list("<1.2.3", &mut process), &mut process),
        process
    );

    assert_eq_in_process!(
        erlang::list_to_pid(
            Term::str_to_char_list("<1.2.3>", &mut process),
            &mut process
        ),
        Term::external_pid(1, 2, 3, &mut process),
        process
    );

    assert_bad_argument!(
        erlang::list_to_pid(
            Term::str_to_char_list("<1.2.3>?", &mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_small_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let small_integer_term = 0.into_process(&mut process);

    assert_bad_argument!(
        erlang::list_to_pid(small_integer_term, &mut process),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(erlang::list_to_pid(big_integer_term, &mut process), process);
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(erlang::list_to_pid(float_term, &mut process), process);
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_bad_argument!(erlang::list_to_pid(local_pid_term, &mut process), process);
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::list_to_pid(external_pid_term, &mut process),
        process
    );
}

#[test]
fn with_tuple_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(erlang::list_to_pid(tuple_term, &mut process), process);
}

#[test]
fn with_heap_binary_is_false() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(erlang::list_to_pid(heap_binary_term, &mut process), process);
}

#[test]
fn with_subbinary_is_false() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_bad_argument!(erlang::list_to_pid(subbinary_term, &mut process), process);
}

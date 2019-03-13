use super::*;

use num_traits::Num;

#[test]
fn with_atom_is_bad_argument() {
    let mut process: Process = Default::default();
    let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

    assert_bad_argument!(
        erlang::insert_element(
            atom_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_empty_list_is_bad_argument() {
    let mut process: Process = Default::default();

    assert_bad_argument!(
        erlang::insert_element(
            Term::EMPTY_LIST,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_list_is_bad_argument() {
    let mut process: Process = Default::default();
    let list_term = list_term(&mut process);

    assert_bad_argument!(
        erlang::insert_element(
            list_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
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
        erlang::insert_element(
            small_integer_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_big_integer_is_bad_argument() {
    let mut process: Process = Default::default();
    let big_integer_term = <BigInt as Num>::from_str_radix("576460752303423489", 10)
        .unwrap()
        .into_process(&mut process);

    assert_bad_argument!(
        erlang::insert_element(
            big_integer_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_float_is_bad_argument() {
    let mut process: Process = Default::default();
    let float_term = 1.0.into_process(&mut process);

    assert_bad_argument!(
        erlang::insert_element(
            float_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_local_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let local_pid_term = Term::local_pid(0, 0).unwrap();

    assert_bad_argument!(
        erlang::insert_element(
            local_pid_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_external_pid_is_bad_argument() {
    let mut process: Process = Default::default();
    let external_pid_term = Term::external_pid(1, 0, 0, &mut process).unwrap();

    assert_bad_argument!(
        erlang::insert_element(
            external_pid_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_tuple_without_small_integer_index_is_bad_argument() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 2.into_process(&mut process)],
        &mut process,
    );
    let index = 1usize;
    let invalid_index_term = Term::arity(index);

    assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
    assert_bad_argument!(
        erlang::insert_element(
            tuple_term,
            invalid_index_term,
            0.into_process(&mut process),
            &mut process
        ),
        process
    );

    let valid_index_term: Term = index.into_process(&mut process);

    assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
    assert_eq_in_process!(
        erlang::insert_element(
            tuple_term,
            valid_index_term,
            1.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[
                0.into_process(&mut process),
                1.into_process(&mut process),
                2.into_process(&mut process)
            ],
            &mut process
        )),
        process
    );
}

#[test]
fn with_tuple_without_index_in_range_is_bad_argument() {
    let mut process: Process = Default::default();
    let empty_tuple_term = Term::slice_to_tuple(&[], &mut process);

    assert_bad_argument!(
        erlang::insert_element(
            empty_tuple_term,
            1.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_tuple_with_index_in_range_returns_tuple_with_new_element_at_index() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(
        &[0.into_process(&mut process), 2.into_process(&mut process)],
        &mut process,
    );

    assert_eq_in_process!(
        erlang::insert_element(
            tuple_term,
            1.into_process(&mut process),
            1.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[
                0.into_process(&mut process),
                1.into_process(&mut process),
                2.into_process(&mut process)
            ],
            &mut process
        )),
        process
    );
}

#[test]
fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
    let mut process: Process = Default::default();
    let tuple_term = Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

    assert_eq_in_process!(
        erlang::insert_element(
            tuple_term,
            1.into_process(&mut process),
            1.into_process(&mut process),
            &mut process
        ),
        Ok(Term::slice_to_tuple(
            &[0.into_process(&mut process), 1.into_process(&mut process)],
            &mut process
        )),
        process
    )
}

#[test]
fn with_heap_binary_is_bad_argument() {
    let mut process: Process = Default::default();
    let heap_binary_term = Term::slice_to_binary(&[], &mut process);

    assert_bad_argument!(
        erlang::insert_element(
            heap_binary_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

#[test]
fn with_subbinary_is_bad_argument() {
    let mut process: Process = Default::default();
    let binary_term =
        Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
    let subbinary_term = Term::subbinary(binary_term, 0, 7, 2, 1, &mut process);

    assert_bad_argument!(
        erlang::insert_element(
            subbinary_term,
            0.into_process(&mut process),
            0.into_process(&mut process),
            &mut process
        ),
        process
    );
}

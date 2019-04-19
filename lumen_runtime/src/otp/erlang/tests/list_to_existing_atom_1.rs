use super::*;

#[test]
fn with_atom_errors_badarg() {
    errors_badarg(|_| Term::str_to_atom("list", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_errors_badarg() {
    errors_badarg(|process| Term::local_reference(&process));
}

#[test]
fn with_empty_list() {
    let list = Term::EMPTY_LIST;

    // as `""` can only be entered into the global atom table, can't test with non-existing atom
    let existing_atom = Term::str_to_atom("", DoNotCare).unwrap();

    assert_eq!(erlang::list_to_existing_atom_1(list), Ok(existing_atom));
}

#[test]
fn with_improper_list_errors_badarg() {
    errors_badarg(|process| {
        Term::cons(
            'a'.into_process(&process),
            'b'.into_process(&process),
            &process,
        )
    });
}

#[test]
fn with_list_encoding_utf8() {
    with_process(|process| {
        let string1 = format!("{}:{}:atom", file!(), line!());
        let char_list1 = Term::str_to_char_list(&string1, &process);

        assert_badarg!(erlang::list_to_existing_atom_1(char_list1));

        let existing_atom1 = Term::str_to_atom(&string1, DoNotCare).unwrap();

        assert_eq!(
            erlang::list_to_existing_atom_1(char_list1),
            Ok(existing_atom1)
        );

        let string2 = format!("{}:{}:José", file!(), line!());
        let char_list2 = Term::str_to_char_list(&string2, &process);

        assert_badarg!(erlang::list_to_existing_atom_1(char_list2));

        let existing_atom2 = Term::str_to_atom(&string2, DoNotCare).unwrap();

        assert_eq!(
            erlang::list_to_existing_atom_1(char_list2),
            Ok(existing_atom2)
        );

        let string3 = format!("{}:{}:😈", file!(), line!());
        let char_list3 = Term::str_to_char_list(&string3, &process);

        assert_badarg!(erlang::list_to_existing_atom_1(char_list3));

        let existing_atom3 = Term::str_to_atom(&string3, DoNotCare).unwrap();

        assert_eq!(
            erlang::list_to_existing_atom_1(char_list3),
            Ok(existing_atom3)
        );
    });
}

#[test]
fn with_list_not_encoding_ut8() {
    errors_badarg(|process| {
        Term::cons(
            // from https://doc.rust-lang.org/std/char/fn.from_u32.html
            0x110000.into_process(&process),
            Term::EMPTY_LIST,
            &process,
        )
    });
}

#[test]
fn with_small_integer_errors_badarg() {
    errors_badarg(|process| 0.into_process(&process));
}

#[test]
fn with_big_integer_errors_badarg() {
    errors_badarg(|process| (integer::small::MAX + 1).into_process(&process));
}

#[test]
fn with_float_errors_badarg() {
    errors_badarg(|process| 1.0.into_process(&process));
}

#[test]
fn with_local_pid_errors_badarg() {
    errors_badarg(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_errors_badarg() {
    errors_badarg(|process| Term::external_pid(1, 0, 0, &process).unwrap());
}

#[test]
fn with_tuple_errors_badarg() {
    errors_badarg(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_errors_badarg() {
    errors_badarg(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_errors_badmap() {
    errors_badarg(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_errors_badmap() {
    errors_badarg(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn errors_badarg<F>(string: F)
where
    F: FnOnce(&Process) -> Term,
{
    super::errors_badarg(|process| erlang::list_to_existing_atom_1(string(&process)));
}

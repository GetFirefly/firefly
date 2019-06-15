use super::*;

mod with_big_integer_augend;
mod with_float_augend;
mod with_small_integer_augend;

#[test]
fn with_atom_augend_errors_badarith() {
    with_augend_errors_badarith(|_| Term::str_to_atom("augend", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_augend_errors_badarith() {
    with_augend_errors_badarith(|process| Term::next_local_reference(process));
}

#[test]
fn with_empty_list_augend_errors_badarith() {
    with_augend_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_augend_errors_badarith() {
    with_augend_errors_badarith(|process| {
        Term::cons(0.into_process(&process), 1.into_process(&process), &process)
    });
}

#[test]
fn with_local_pid_augend_errors_badarith() {
    with_augend_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_augend_errors_badarith() {
    with_augend_errors_badarith(|process| Term::external_pid(1, 2, 3, &process).unwrap());
}

#[test]
fn with_tuple_augend_errors_badarith() {
    with_augend_errors_badarith(|process| Term::slice_to_tuple(&[], &process));
}

#[test]
fn with_map_is_augend_errors_badarith() {
    with_augend_errors_badarith(|process| Term::slice_to_map(&[], &process));
}

#[test]
fn with_heap_binary_augend_errors_badarith() {
    with_augend_errors_badarith(|process| Term::slice_to_binary(&[], &process));
}

#[test]
fn with_subbinary_augend_errors_badarith() {
    with_augend_errors_badarith(|process| {
        let original = Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &process);
        Term::subbinary(original, 0, 7, 2, 1, &process)
    });
}

fn with_augend_errors_badarith<M>(augend: M)
where
    M: FnOnce(&Process) -> Term,
{
    super::errors_badarith(|process| {
        let augend = augend(&process);
        let addend = 0.into_process(&process);

        erlang::add_2(augend, addend, &process)
    });
}

use super::*;

mod with_big_integer_multiplier;
mod with_float_multiplier;
mod with_small_integer_multiplier;

#[test]
fn with_atom_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|_| Term::str_to_atom("multiplier", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| {
        Term::cons(
            0.into_process(&mut process),
            1.into_process(&mut process),
            &mut process,
        )
    });
}

#[test]
fn with_local_pid_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|_| Term::local_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| {
        Term::external_pid(1, 2, 3, &mut process).unwrap()
    });
}

#[test]
fn with_tuple_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_is_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| Term::slice_to_binary(&[], &mut process));
}

#[test]
fn with_subbinary_multiplier_errors_badarith() {
    with_multiplier_errors_badarith(|mut process| {
        let original =
            Term::slice_to_binary(&[0b0000_00001, 0b1111_1110, 0b1010_1011], &mut process);
        Term::subbinary(original, 0, 7, 2, 1, &mut process)
    });
}

fn with_multiplier_errors_badarith<M>(multiplier: M)
where
    M: FnOnce(&mut Process) -> Term,
{
    super::errors_badarith(|mut process| {
        let multiplier = multiplier(&mut process);
        let multiplicand = 0.into_process(&mut process);

        erlang::multiply_2(multiplier, multiplicand, &mut process)
    });
}

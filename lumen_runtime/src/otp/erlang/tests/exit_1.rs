use super::*;

use num_traits::Num;

use crate::process::IntoProcess;

#[test]
fn with_atom_exits_with_atom_reason() {
    exits(|_| Term::str_to_atom("reason", DoNotCare).unwrap());
}

#[test]
fn with_local_reference_exits_with_local_reference() {
    exits(|mut process| Term::local_reference(&mut process));
}

#[test]
fn with_empty_list_exits_with_empty_list_reason() {
    exits(|_| Term::EMPTY_LIST);
}

#[test]
fn with_list_exits_with_list_reason() {
    exits(|mut process| list_term(&mut process));
}

#[test]
fn with_small_integer_exits_with_small_integer_reason() {
    exits(|mut process| 0usize.into_process(&mut process));
}

#[test]
fn with_big_integer_exits_with_big_integer_reason() {
    exits(|mut process| {
        <BigInt as Num>::from_str_radix("576460752303423489", 10)
            .unwrap()
            .into_process(&mut process)
    });
}

#[test]
fn with_float_exits_with_float_reason() {
    exits(|mut process| 1.0.into_process(&mut process));
}

#[test]
fn with_local_pid_exits_with_local_pid_reason() {
    exits(|_| Term::local_pid(0, 0).unwrap());
}

#[test]
fn with_external_pid_exits_with_external_pid_reason() {
    exits(|mut process| Term::external_pid(1, 0, 0, &mut process).unwrap());
}

#[test]
fn with_tuple_exits_with_tuple_reason() {
    exits(|mut process| Term::slice_to_tuple(&[], &mut process));
}

#[test]
fn with_map_exits_with_map_reason() {
    exits(|mut process| Term::slice_to_map(&[], &mut process));
}

#[test]
fn with_heap_binary_exits_with_heap_binary_reason() {
    exits(|mut process| {
        // :erlang.term_to_binary(:atom)
        Term::slice_to_binary(&[131, 100, 0, 4, 97, 116, 111, 109], &mut process)
    });
}

#[test]
fn with_subbinary_exits_with_subbinary_reason() {
    exits(|mut process| {
        // <<1::1, :erlang.term_to_binary(:atom) :: binary>>
        let original_term = Term::slice_to_binary(
            &[193, 178, 0, 2, 48, 186, 55, 182, 0b1000_0000],
            &mut process,
        );
        Term::subbinary(original_term, 0, 1, 8, 0, &mut process)
    });
}

fn exits<R>(reason: R)
where
    R: FnOnce(&mut Process) -> Term,
{
    with_process(|mut process| {
        let reason = reason(&mut process);

        assert_exit!(erlang::exit_1(reason), reason);
    })
}

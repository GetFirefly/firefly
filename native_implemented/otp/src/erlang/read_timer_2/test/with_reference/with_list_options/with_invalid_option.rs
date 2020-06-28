use super::*;

#[test]
fn without_reference_errors_badarg() {
    run!(
        |arc_process| {
            (
                Just(arc_process.clone()),
                strategy::term::is_not_reference(arc_process.clone()),
                options(arc_process.clone()),
            )
        },
        |(arc_process, timer_reference, options)| {
            prop_assert_badarg!(
                result(&arc_process, timer_reference, options),
                format!(
                    "timer_reference ({}) is not a local reference",
                    timer_reference
                )
            );

            Ok(())
        },
    );
}

#[test]
fn with_atom_option_errors_badarg() {
    with_option_errors_badarg(|_| Atom::str_to_term("unsupported"));
}

#[test]
fn with_local_reference_option_errors_badarg() {
    with_option_errors_badarg(|process| process.next_reference().unwrap());
}

#[test]
fn with_small_integer_option_errors_badarg() {
    with_option_errors_badarg(|process| process.integer(0).unwrap());
}

#[test]
fn with_big_integer_option_errors_badarg() {
    with_option_errors_badarg(|process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap());
}

#[test]
fn with_float_option_errors_badarg() {
    with_option_errors_badarg(|process| process.float(1.0).unwrap());
}

#[test]
fn with_local_pid_option_errors_badarg() {
    with_option_errors_badarg(|_| Pid::make_term(0, 1).unwrap());
}

#[test]
fn with_external_pid_option_errors_badarg() {
    with_option_errors_badarg(|process| process.external_pid(external_arc_node(), 2, 3).unwrap());
}

#[test]
fn with_tuple_option_errors_badarg() {
    with_option_errors_badarg(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_option_errors_badarg() {
    with_option_errors_badarg(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_empty_list_option_errors_badarg() {
    with_option_errors_badarg(|process| {
        process
            .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
            .unwrap()
    });
}

#[test]
fn with_list_option_errors_badarg() {
    with_option_errors_badarg(|process| {
        process
            .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
            .unwrap()
    });
}

#[test]
fn with_heap_binary_option_errors_badarg() {
    with_option_errors_badarg(|process| process.binary_from_bytes(&[]).unwrap());
}

#[test]
fn with_subbinary_option_errors_badarg() {
    with_option_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_option_errors_badarg<O>(option: O)
where
    O: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let timer_reference = process.next_reference().unwrap();
        let options = process.cons(option(process), Term::NIL).unwrap();

        assert_badarg!(
            result(process, timer_reference, options),
            "supported option is {:async, bool}"
        );
    });
}

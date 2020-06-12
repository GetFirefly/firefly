use super::*;

#[test]
fn with_atom_right_returns_true() {
    with_right_returns_right(|mut _process| Atom::str_to_term("right"));
}

#[test]
fn with_false_right_returns_true() {
    with_right_returns_right(|_| false.into());
}

#[test]
fn with_true_right_returns_true() {
    with_right_returns_right(|_| true.into());
}

#[test]
fn with_local_reference_right_returns_true() {
    with_right_returns_right(|process| process.next_reference().unwrap());
}

#[test]
fn with_empty_list_right_returns_true() {
    with_right_returns_right(|_| Term::NIL);
}

#[test]
fn with_list_right_returns_true() {
    with_right_returns_right(|process| {
        process
            .cons(process.integer(0).unwrap(), process.integer(1).unwrap())
            .unwrap()
    });
}

#[test]
fn with_small_integer_right_returns_true() {
    with_right_returns_right(|process| process.integer(1).unwrap())
}

#[test]
fn with_big_integer_right_returns_true() {
    with_right_returns_right(|process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap())
}

#[test]
fn with_float_right_returns_true() {
    with_right_returns_right(|process| process.float(1.0).unwrap());
}

#[test]
fn with_local_pid_right_returns_true() {
    with_right_returns_right(|_| Pid::make_term(0, 1).unwrap());
}

#[test]
fn with_external_pid_right_returns_true() {
    with_right_returns_right(|process| process.external_pid(external_arc_node(), 2, 3).unwrap());
}

#[test]
fn with_tuple_right_returns_true() {
    with_right_returns_right(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_is_right_returns_true() {
    with_right_returns_right(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_heap_binary_right_returns_true() {
    with_right_returns_right(|process| process.binary_from_bytes(&[]).unwrap());
}

#[test]
fn with_subbinary_right_returns_true() {
    with_right_returns_right(|process| bitstring!(1 :: 1, &process));
}

fn with_right_returns_right<R>(right: R)
where
    R: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let left = false.into();
        let right = right(&process);

        assert_eq!(result(left, right), Ok(right));
    });
}

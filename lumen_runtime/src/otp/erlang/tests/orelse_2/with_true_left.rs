use super::*;

#[test]
fn with_atom_right_returns_true() {
    with_right_returns_true(|mut _process| atom_unchecked("right"));
}

#[test]
fn with_false_right_returns_true() {
    with_right_returns_true(|_| false.into());
}

#[test]
fn with_true_right_returns_true() {
    with_right_returns_true(|_| true.into());
}

#[test]
fn with_local_reference_right_returns_true() {
    with_right_returns_true(|process| process.next_reference().unwrap());
}

#[test]
fn with_empty_list_right_returns_true() {
    with_right_returns_true(|_| Term::NIL);
}

#[test]
fn with_list_right_returns_true() {
    with_right_returns_true(|process| {
        let mut heap = process.acquire_heap();
        heap.cons(heap.integer(0).unwrap(), heap.integer(1).unwrap())
            .unwrap()
            .unwrap();
    });
}

#[test]
fn with_small_integer_right_returns_true() {
    with_right_returns_true(|process| process.integer(1).unwrap())
}

#[test]
fn with_big_integer_right_returns_true() {
    with_right_returns_true(|process| process.integer(SmallInteger::MAX_VALUE + 1).unwrap())
}

#[test]
fn with_float_right_returns_true() {
    with_right_returns_true(|process| process.float(1.0).unwrap());
}

#[test]
fn with_local_pid_right_returns_true() {
    with_right_returns_true(|_| make_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_right_returns_true() {
    with_right_returns_true(|process| process.external_pid_with_node_id(1, 2, 3).unwrap());
}

#[test]
fn with_tuple_right_returns_true() {
    with_right_returns_true(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_is_right_returns_true() {
    with_right_returns_true(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_heap_binary_right_returns_true() {
    with_right_returns_true(|process| process.binary_from_bytes(&[]).unwrap());
}

#[test]
fn with_subbinary_right_returns_true() {
    with_right_returns_true(|process| bitstring!(1 :: 1, &process));
}

fn with_right_returns_true<R>(right: R)
where
    R: FnOnce(&Process) -> Term,
{
    with_process(|process| {
        let left = true.into();
        let right = right(&process);

        assert_eq!(erlang::orelse_2(left, right), Ok(left));
    });
}

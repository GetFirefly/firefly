use super::*;

mod with_atom_name;

#[test]
fn without_atom_name_errors_badarg() {
    with_process_arc(|arc_process| {
        TestRunner::new(Config::with_source_file(file!()))
            .run(
                &(
                    strategy::term::is_not_atom(arc_process.clone()),
                    strategy::term(arc_process.clone()),
                ),
                |(name, message)| {
                    let destination = arc_process
                        .tuple_from_slice(&[name, erlang::node_0()])
                        .unwrap();

                    prop_assert_eq!(
                        erlang::send_2(destination, message, &arc_process),
                        Err(badarg!().into())
                    );

                    Ok(())
                },
            )
            .unwrap();
    });
}

#[test]
fn with_local_reference_name_errors_badarg() {
    with_name_errors_badarg(|process| process.next_reference().unwrap());
}

#[test]
fn with_empty_list_name_errors_badarg() {
    with_name_errors_badarg(|_| Term::NIL);
}

#[test]
fn with_list_name_errors_badarg() {
    with_name_errors_badarg(|process| {
        process
            .cons(process.integer(0), process.integer(1))
            .unwrap()
    });
}

#[test]
fn with_small_integer_name_errors_badarg() {
    with_name_errors_badarg(|process| process.integer(0));
}

#[test]
fn with_big_integer_name_errors_badarg() {
    with_name_errors_badarg(|process| process.integer(SmallInteger::MAX_VALUE + 1));
}

#[test]
fn with_float_name_errors_badarg() {
    with_name_errors_badarg(|process| process.float(1.0).unwrap());
}

#[test]
fn with_local_pid_name_errors_badarg() {
    with_name_errors_badarg(|_| make_pid(0, 1).unwrap());
}

#[test]
fn with_external_pid_name_errors_badarg() {
    with_name_errors_badarg(|process| process.external_pid_with_node_id(1, 2, 3).unwrap());
}

#[test]
fn with_tuple_name_errors_badarg() {
    with_name_errors_badarg(|process| process.tuple_from_slice(&[]).unwrap());
}

#[test]
fn with_map_name_errors_badarg() {
    with_name_errors_badarg(|process| process.map_from_slice(&[]).unwrap());
}

#[test]
fn with_heap_binary_name_errors_badarg() {
    with_name_errors_badarg(|process| process.binary_from_bytes(&[]).unwrap());
}

#[test]
fn with_subbinary_name_errors_badarg() {
    with_name_errors_badarg(|process| bitstring!(1 :: 1, &process));
}

fn with_name_errors_badarg<N>(name: N)
where
    N: FnOnce(&ProcessControlBlock) -> Term,
{
    with_process(|process| {
        let destination = process
            .tuple_from_slice(&[name(process), erlang::node_0()])
            .unwrap();
        let message = atom_unchecked("message");

        assert_badarg!(erlang::send_2(destination, message, process));
    })
}

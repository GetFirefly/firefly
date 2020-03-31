use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler::Spawned;

use crate::test;

#[test]
fn without_heap_available_does_not_modify_dictionary() {
    crate::test::process_dictionary::without_heap_available_does_not_modify_dictionary(native);
}

#[test]
fn with_heap_available_returns_entries_as_list() {
    let init_arc_process = test::process::init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let key = Atom::str_to_term("key");
    let value = Atom::str_to_term("value");

    arc_process.put(key, value).unwrap();

    assert_eq!(arc_process.get_value_from_key(key), value);

    let result = native(&arc_process);

    assert!(result.is_ok());

    let list = result.unwrap();

    assert!(list.is_list());

    let boxed_cons: Boxed<Cons> = list.try_into().unwrap();
    let vec: Vec<Term> = boxed_cons
        .into_iter()
        .map(|result| result.unwrap())
        .collect();

    assert_eq!(vec.len(), 1);
    assert!(vec.contains(&key));
}

// From https://github.com/erlang/otp/blob/a62aed81c56c724f7dd7040adecaa28a78e5d37f/erts/doc/src/erlang.xml#L2089-L2094
#[test]
fn doc_test() {
    let init_arc_process = test::process::init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let animal = Atom::str_to_term("animal");

    let dog = Atom::str_to_term("dog");
    arc_process
        .put(
            dog,
            arc_process
                .tuple_from_slice(&[animal, arc_process.integer(1).unwrap()])
                .unwrap(),
        )
        .unwrap();

    let cow = Atom::str_to_term("cow");
    arc_process
        .put(
            cow,
            arc_process
                .tuple_from_slice(&[animal, arc_process.integer(2).unwrap()])
                .unwrap(),
        )
        .unwrap();

    let lamb = Atom::str_to_term("lamb");
    arc_process
        .put(
            lamb,
            arc_process
                .tuple_from_slice(&[animal, arc_process.integer(3).unwrap()])
                .unwrap(),
        )
        .unwrap();

    let result = native(&arc_process);

    assert!(result.is_ok());

    let list = result.unwrap();

    assert!(list.is_list());

    let boxed_cons: Boxed<Cons> = list.try_into().unwrap();
    let vec: Vec<Term> = boxed_cons
        .into_iter()
        .map(|result| result.unwrap())
        .collect();

    assert_eq!(vec.len(), 3);
    assert!(vec.contains(&dog));
    assert!(vec.contains(&cow));
    assert!(vec.contains(&lamb));
}

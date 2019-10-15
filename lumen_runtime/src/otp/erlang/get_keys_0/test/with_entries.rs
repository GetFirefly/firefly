use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Cons};

use crate::process;
use crate::scheduler::Spawned;

#[test]
fn without_heap_available_does_not_modify_dictionary() {
    let init_arc_process = process::test_init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let key = atom_unchecked("key");
    let value = atom_unchecked("value");

    arc_process.put(key, value).unwrap();

    fill_heap(&arc_process);

    assert_eq!(arc_process.get_value_from_key(key), value);

    assert_eq!(native(&arc_process), Err(liblumen_alloc::alloc!().into()));

    assert_eq!(arc_process.get_value_from_key(key), value);
}

#[test]
fn with_heap_available_returns_entries_as_list() {
    let init_arc_process = process::test_init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let key = atom_unchecked("key");
    let value = atom_unchecked("value");

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
    let init_arc_process = process::test_init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let animal = atom_unchecked("animal");

    let dog = atom_unchecked("dog");
    arc_process
        .put(
            dog,
            arc_process
                .tuple_from_slice(&[animal, arc_process.integer(1).unwrap()])
                .unwrap(),
        )
        .unwrap();

    let cow = atom_unchecked("cow");
    arc_process
        .put(
            cow,
            arc_process
                .tuple_from_slice(&[animal, arc_process.integer(2).unwrap()])
                .unwrap(),
        )
        .unwrap();

    let lamb = atom_unchecked("lamb");
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

fn fill_heap(process: &Process) {
    {
        let mut heap = process.acquire_heap();

        while let Ok(_) = heap.cons(atom_unchecked("hd"), atom_unchecked("tl")) {}
    }
}

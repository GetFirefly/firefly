use super::*;

use std::convert::TryInto;

use liblumen_alloc::erts::process::alloc::heap_alloc::HeapAlloc;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::{atom_unchecked, Boxed, Cons, Tuple};

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
fn with_heap_available_erases_dictionary_and_returns_entries_as_list() {
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

    let head = boxed_cons.head;

    assert!(head.is_tuple());

    let head_boxed_tuple: Boxed<Tuple> = head.try_into().unwrap();

    assert_eq!(head_boxed_tuple.len(), 2);

    assert_eq!(head_boxed_tuple[0], key);
    assert_eq!(head_boxed_tuple[1], value);

    assert_eq!(boxed_cons.tail, Term::NIL);

    assert_eq!(
        arc_process.get_value_from_key(key),
        atom_unchecked("undefined")
    );
}

fn fill_heap(process: &Process) {
    {
        let mut heap = process.acquire_heap();

        while let Ok(_) = heap.cons(atom_unchecked("hd"), atom_unchecked("tl")) {}
    }
}

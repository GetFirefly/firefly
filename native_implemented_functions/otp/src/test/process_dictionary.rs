use super::*;

use liblumen_alloc::erts::process::alloc::TermAlloc;

use crate::runtime::scheduler::Spawned;

pub fn fill_heap(process: &Process) {
    {
        let mut heap = process.acquire_heap();

        while let Ok(_) = heap.cons(Atom::str_to_term("hd"), Atom::str_to_term("tl")) {}
    }
}

pub fn without_heap_available_does_not_modify_dictionary(
    native: fn(&Process) -> exception::Result<Term>,
) {
    let init_arc_process = process::init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let key = Atom::str_to_term("key");
    let value = Atom::str_to_term("value");

    arc_process.put(key, value).unwrap();

    fill_heap(&arc_process);

    assert_eq!(arc_process.get_value_from_key(key), value);

    assert_eq!(native(&arc_process), Err(liblumen_alloc::alloc!().into()));

    assert_eq!(arc_process.get_value_from_key(key), value);
}

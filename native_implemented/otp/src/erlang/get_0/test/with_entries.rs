use super::*;

use liblumen_alloc::erts::term::prelude::*;

use crate::runtime::scheduler::Spawned;

use crate::test;

#[test]
fn without_heap_available_errors_alloc() {
    let init_arc_process = test::process::init();
    let Spawned { arc_process, .. } = crate::test::process(&init_arc_process, Default::default());
    let key = Atom::str_to_term("key");
    let value = Atom::str_to_term("value");

    arc_process.put(key, value).unwrap();

    crate::test::process_dictionary::fill_heap(&arc_process);

    assert_eq!(arc_process.get_value_from_key(key), value);

    assert_eq!(result(&arc_process), Err(liblumen_alloc::alloc!().into()));
}

// `returns_entries_as_list` in integration tests

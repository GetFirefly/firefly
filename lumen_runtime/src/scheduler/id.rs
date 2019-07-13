// Based on https://github.com/Amanieu/thread_local-rs/blob/8c956ed8642175f1a3afc409bf5f8844d3ea994a/src/thread_id.rs
use std::collections::BinaryHeap;

pub use liblumen_alloc::erts::scheduler::id::Raw;

// Manager which allocates scheduler IDs. It attempts to aggressively reuse scheduler IDs to allow
// scheduler ID to keep inside the number of expected concurrent schedulers and not the total number
// of schedulers spawned in a `cargo test` run, which is greater than or equal to the number of
// tests as each test is run in a separate thread and some threads spawn their own threads to test
// inter-schedule communication.
pub struct Manager {
    next: Raw,
    free_heap: BinaryHeap<Raw>,
}

impl Manager {
    pub fn new() -> Manager {
        Manager {
            next: 0,
            free_heap: BinaryHeap::new(),
        }
    }

    pub fn alloc(&mut self) -> Raw {
        match self.free_heap.pop() {
            Some(id) => id,
            None => {
                let id = self.next;
                self.next = self.next.checked_add(1).expect("Scheduler ID overflow");

                id
            }
        }
    }

    pub fn free(&mut self, id: Raw) {
        self.free_heap.push(id)
    }
}

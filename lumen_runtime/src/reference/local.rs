use std::hash::{Hash, Hasher};
use std::sync::Arc;

use crate::heap::{CloneIntoHeap, Heap};
use crate::scheduler::{self, Scheduler};
use crate::term::{Tag::LocalReference, Term};

pub type Number = u64;

pub struct Reference {
    #[allow(dead_code)]
    header: Term,
    scheduler_id: scheduler::ID,
    number: Number,
}

impl Reference {
    pub fn new(scheduler_id: &scheduler::ID, number: Number) -> Reference {
        Reference {
            header: Term {
                tagged: LocalReference as usize,
            },
            scheduler_id: scheduler_id.clone(),
            number,
        }
    }

    pub fn number(&self) -> Number {
        self.number
    }

    pub fn scheduler(&self) -> Option<Arc<Scheduler>> {
        Scheduler::from_id(&self.scheduler_id)
    }

    pub fn scheduler_id(&self) -> scheduler::ID {
        self.scheduler_id.clone()
    }
}

impl CloneIntoHeap for &'static Reference {
    fn clone_into_heap(&self, heap: &Heap) -> &'static Reference {
        heap.local_reference(&self.scheduler_id, self.number)
    }
}

impl Eq for Reference {}

impl Hash for Reference {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.number.hash(state);
    }
}

impl PartialEq for Reference {
    fn eq(&self, other: &Reference) -> bool {
        self.number == other.number
    }
}

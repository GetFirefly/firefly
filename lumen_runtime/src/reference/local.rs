use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::heap::{CloneIntoHeap, Heap};
use crate::term::{Tag::LocalReference, Term};

pub type Number = u64;

pub struct Reference {
    #[allow(dead_code)]
    header: Term,
    number: Number,
}

impl Reference {
    pub fn new(number: Number) -> Reference {
        Reference {
            header: Term {
                tagged: LocalReference as usize,
            },
            number,
        }
    }

    pub fn next() -> Reference {
        Self::new(COUNT.fetch_add(1, Ordering::SeqCst))
    }

    pub fn number(&self) -> Number {
        self.number
    }
}

impl CloneIntoHeap for &'static Reference {
    fn clone_into_heap(&self, heap: &Heap) -> &'static Reference {
        heap.u64_to_local_reference(self.number)
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

// References are always 64-bits even on 32-bit platforms
static COUNT: AtomicU64 = AtomicU64::new(0);

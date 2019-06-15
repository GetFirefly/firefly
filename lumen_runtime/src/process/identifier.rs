use std::cmp::Ordering::{self, *};
use std::hash::{Hash, Hasher};

use crate::heap::{CloneIntoHeap, Heap};
use crate::term::{Tag, Tag::*, Term};

pub mod local;

pub const NUMBER_BIT_COUNT: u8 = 15;
pub const NUMBER_MASK: usize = 0b111_1111_1111_1111;
pub const NUMBER_MAX: usize = (1 << (NUMBER_BIT_COUNT as usize)) - 1;

const SERIAL_BIT_COUNT: u8 = Term::BIT_COUNT - NUMBER_BIT_COUNT - Tag::LOCAL_PID_BIT_COUNT;
pub const SERIAL_MAX: usize = (1 << (SERIAL_BIT_COUNT as usize)) - 1;

pub struct External {
    #[allow(dead_code)]
    header: Term,
    pub node: usize,
    pub serial: usize,
    pub number: usize,
}

impl External {
    pub fn new(node: usize, number: usize, serial: usize) -> Self {
        assert_ne!(node, 0, "Node 0 is reserved for the local node");

        External {
            header: Term {
                tagged: (ExternalPid as usize),
            },
            node,
            serial,
            number,
        }
    }
}

impl CloneIntoHeap for &'static External {
    fn clone_into_heap(&self, heap: &Heap) -> &'static External {
        heap.external_pid(self.node, self.number, self.serial)
    }
}

impl Eq for External {}

impl Hash for External {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.serial.hash(state);
        self.number.hash(state);
    }
}

impl Ord for External {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.node.cmp(&other.node) {
            Equal => match self.serial.cmp(&other.serial) {
                Equal => self.number.cmp(&other.number),
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

impl PartialEq for External {
    fn eq(&self, other: &External) -> bool {
        self.cmp(other) == Equal
    }
}

impl PartialOrd for External {
    fn partial_cmp(&self, other: &External) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

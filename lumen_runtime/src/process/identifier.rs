use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use crate::process::{OrderInProcess, Process};
use crate::term::{Tag, Tag::*, Term};

pub const NUMBER_BIT_COUNT: u8 = 15;
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

impl Hash for External {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        self.serial.hash(state);
        self.number.hash(state);
    }
}

impl OrderInProcess for External {
    fn cmp_in_process(&self, other: &External, _process: &Process) -> Ordering {
        match self.node.cmp(&other.node) {
            Ordering::Equal => match self.serial.cmp(&other.serial) {
                Ordering::Equal => self.number.cmp(&other.number),
                ordering => ordering,
            },
            ordering => ordering,
        }
    }
}

impl PartialEq for External {
    fn eq(&self, other: &External) -> bool {
        (self.node == other.node) & (self.serial == other.serial) & (self.number == other.number)
    }

    fn ne(&self, other: &External) -> bool {
        !self.eq(other)
    }
}

pub struct LocalCounter {
    serial: usize,
    number: usize,
}

impl LocalCounter {
    pub fn next(&mut self) -> Term {
        let local_pid = unsafe { Term::local_pid_unchecked(self.number, self.serial) };

        if NUMBER_MAX <= self.number {
            self.serial += 1;
            self.number = 0;
        } else {
            self.number += 1;
        }

        local_pid
    }
}

impl Default for LocalCounter {
    fn default() -> LocalCounter {
        LocalCounter {
            serial: 0,
            number: 0,
        }
    }
}

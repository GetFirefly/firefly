use std::cmp::Ordering;

use crate::process::{OrderInProcess, Process};
use crate::term::{Tag, Term};

pub const NUMBER_BIT_COUNT: u8 = 15;
pub const NUMBER_MAX: usize = (1 << (NUMBER_BIT_COUNT as usize)) - 1;

const SERIAL_BIT_COUNT: u8 = Term::BIT_COUNT - NUMBER_BIT_COUNT - Tag::LOCAL_PID_BIT_COUNT;
pub const SERIAL_MAX: usize = (1 << (SERIAL_BIT_COUNT as usize)) - 1;

pub struct External {
    #[allow(dead_code)]
    header: Term,
    node: usize,
    serial: usize,
    number: usize,
}

impl External {
    pub fn new(node: usize, number: usize, serial: usize) -> Self {
        assert_ne!(node, 0, "Node 0 is reserved for the local node");

        External {
            header: Term {
                tagged: (Tag::ExternalPid as usize),
            },
            node,
            serial,
            number,
        }
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

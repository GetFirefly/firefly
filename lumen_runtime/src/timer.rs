use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::ops::{Index, IndexMut};
use std::rc::{self, Rc};
use std::sync;

use crate::atom::Existence::DoNotCare;
use crate::heap::{self, CloneIntoHeap};
use crate::process::Process;
use crate::reference;
use crate::registry::{self, Registered};
use crate::term::Term;
use crate::time::monotonic::{self, Milliseconds};

pub fn start(
    monotonic_time_milliseconds: Milliseconds,
    destination: Destination,
    process_message: Term,
    process: &Process,
) -> Term {
    let reference = HIERARCHY.with(|thread_local_hierarchy| {
        thread_local_hierarchy.borrow_mut().start(
            monotonic_time_milliseconds,
            destination,
            process_message,
            process,
        )
    });

    Term::box_reference(reference)
}

/// Times out the timers for the thread that have timed out since the last time `timeout` was
/// called.
pub fn timeout() {
    HIERARCHY.with(|thread_local_hierarchy| thread_local_hierarchy.borrow_mut().timeout());
}

#[derive(Clone, Debug)]
pub enum Destination {
    Name(Term),
    Process(sync::Weak<Process>),
}

#[derive(Debug)]
struct Timer {
    // Can't be a `Boxed` `LocalReference` `Term` because those are boxed and the original Process
    // could GC the unboxed `LocalReference` `Term`.
    reference_number: u64,
    monotonic_time_milliseconds: Milliseconds,
    destination: Destination,
    #[allow(dead_code)]
    heap: heap::Heap,
    message: Term,
    position: Position,
}

impl Timer {
    fn timeout(self) {
        match &self.destination {
            Destination::Name(ref name) => {
                let readable_registry = registry::RW_LOCK_REGISTERED_BY_NAME.read().unwrap();

                match readable_registry.get(name) {
                    Some(Registered::Process(destination_process_arc)) => {
                        let timeout_message = self.timeout_message();

                        destination_process_arc.send_heap_message(self.heap, timeout_message);
                    }
                    None => (),
                }
            }
            Destination::Process(destination_process_weak) => {
                if let Some(destination_process_arc) = destination_process_weak.upgrade() {
                    let timeout_message = self.timeout_message();

                    destination_process_arc.send_heap_message(self.heap, timeout_message);
                }
            }
        }
    }

    fn timeout_message(&self) -> Term {
        let reference = self.heap.u64_to_local_reference(self.reference_number);
        let reference_term = Term::box_reference(reference);

        let tuple = self.heap.slice_to_tuple(&[
            Term::str_to_atom("timeout", DoNotCare).unwrap(),
            reference_term,
            self.message,
        ]);

        Term::box_reference(tuple)
    }
}

impl Clone for Timer {
    fn clone(&self) -> Timer {
        let cloned_heap: heap::Heap = Default::default();
        let cloned_heap_message = self.message.clone_into_heap(&cloned_heap);

        Timer {
            reference_number: self.reference_number.clone(),
            monotonic_time_milliseconds: self.monotonic_time_milliseconds.clone(),
            destination: self.destination.clone(),
            heap: cloned_heap,
            message: cloned_heap_message,
            position: self.position.clone(),
        }
    }
}

impl Eq for Timer {}

impl PartialEq<Timer> for Timer {
    fn eq(&self, other: &Timer) -> bool {
        self.reference_number == other.reference_number
    }
}

impl Ord for Timer {
    fn cmp(&self, other: &Timer) -> Ordering {
        // Timers are ordered in reverse order as `BinaryHeap` is a max heap, but we want sooner
        // timers at the top
        other
            .monotonic_time_milliseconds
            .cmp(&self.monotonic_time_milliseconds)
            .then_with(|| other.reference_number.cmp(&self.reference_number))
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Timer) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg_attr(test, derive(Debug))]
struct Hierarchy {
    at_once: BinaryHeap<Rc<Timer>>,
    soon: Wheel,
    later: Wheel,
    long_term: BinaryHeap<Rc<Timer>>,
    timer_by_reference_number: HashMap<reference::local::Number, rc::Weak<Timer>>,
}

impl Hierarchy {
    const SOON_MILLISECONDS_PER_SLOT: Milliseconds = 1;
    const SOON_TOTAL_MILLISECONDS: Milliseconds =
        Self::SOON_MILLISECONDS_PER_SLOT * (Wheel::LENGTH as Milliseconds);
    const LATER_MILLISECONDS_PER_SLOT: Milliseconds = Self::SOON_TOTAL_MILLISECONDS / 2;
    const LATER_TOTAL_MILLISECONDS: Milliseconds =
        Self::LATER_MILLISECONDS_PER_SLOT * (Wheel::LENGTH as Milliseconds);

    fn position(&self, monotonic_time_milliseconds: Milliseconds) -> Position {
        if monotonic_time_milliseconds < self.soon.slot_monotonic_time_milliseconds {
            Position::AtOnce
        } else if monotonic_time_milliseconds < self.later.slot_monotonic_time_milliseconds {
            Position::Soon {
                slot_index: self.soon.slot_index(monotonic_time_milliseconds),
            }
        } else if monotonic_time_milliseconds
            < (self.later.slot_monotonic_time_milliseconds + Self::LATER_TOTAL_MILLISECONDS)
        {
            Position::Later {
                slot_index: self.later.slot_index(monotonic_time_milliseconds),
            }
        } else {
            Position::LongTerm
        }
    }

    fn start(
        &mut self,
        monotonic_time_milliseconds: Milliseconds,
        destination: Destination,
        process_message: Term,
        process: &Process,
    ) -> &'static reference::local::Reference {
        let reference = process.local_reference();
        let reference_number = reference.number();
        let heap: heap::Heap = Default::default();
        let heap_message = process_message.clone_into_heap(&heap);
        let position = self.position(monotonic_time_milliseconds);

        let timer = Timer {
            reference_number,
            monotonic_time_milliseconds,
            destination,
            heap,
            message: heap_message,
            position,
        };
        let timer_rc = Rc::new(timer);
        let timeoutable = Rc::clone(&timer_rc);
        let cancellable = Rc::downgrade(&timer_rc);

        match position {
            Position::AtOnce => self.at_once.push(timeoutable),
            Position::Soon { slot_index } => self.soon[slot_index].push(timeoutable),
            Position::Later { slot_index } => self.later[slot_index].push(timeoutable),
            Position::LongTerm => self.long_term.push(timeoutable),
        }

        self.timer_by_reference_number
            .insert(reference_number, cancellable);

        reference
    }

    fn timeout(&mut self) {
        self.timeout_at_once();

        let monotonic_time_milliseconds = monotonic::time_in_milliseconds();
        let milliseconds = monotonic_time_milliseconds - self.soon.slot_monotonic_time_milliseconds;

        for _ in 0..milliseconds {
            self.timeout_soon_slot();

            assert!(self.soon.peek().is_none());
            self.soon.next_slot();

            let soon_max_monotonic_time_milliseconds = self.soon.max_monotonic_time_milliseconds();

            if self.later.slot_monotonic_time_milliseconds <= soon_max_monotonic_time_milliseconds {
                self.transfer_later_to_soon(soon_max_monotonic_time_milliseconds);

                let later_next_slot_monotonic_time_milliseconds =
                    self.later.next_slot_monotonic_time_milliseconds();

                if later_next_slot_monotonic_time_milliseconds
                    <= soon_max_monotonic_time_milliseconds
                {
                    assert!(self.later.peek().is_none());
                    self.later.next_slot();

                    let later_max_monotonic_time_milliseconds =
                        self.later.max_monotonic_time_milliseconds();

                    self.transfer_long_term_to_later(later_max_monotonic_time_milliseconds);
                }
            }
        }
    }

    fn timeout_at_once(&mut self) {
        while let Some(timer_rc) = self.at_once.pop() {
            self.timer_by_reference_number
                .remove(&timer_rc.reference_number);

            Rc::try_unwrap(timer_rc).unwrap().timeout()
        }
    }

    fn timeout_soon_slot(&mut self) {
        while let Some(timer_rc) = self.soon.pop() {
            self.timer_by_reference_number
                .remove(&timer_rc.reference_number);

            Rc::try_unwrap(timer_rc).unwrap().timeout()
        }
    }

    fn transfer(&mut self, mut transferable_timer_rc: Rc<Timer>, wheel_name: WheelName) {
        let wheel = match wheel_name {
            WheelName::Soon => &mut self.soon,
            WheelName::Later => &mut self.later,
        };

        let slot_index = wheel.slot_index(transferable_timer_rc.monotonic_time_milliseconds);

        let position = match wheel_name {
            WheelName::Soon => Position::Soon { slot_index },
            WheelName::Later => Position::Later { slot_index },
        };

        // Remove weak reference to allow `get_mut`.
        let reference_number = transferable_timer_rc.reference_number;
        self.timer_by_reference_number.remove(&reference_number);

        Rc::get_mut(&mut transferable_timer_rc).unwrap().position = position;

        let timeoutable = Rc::clone(&transferable_timer_rc);
        let cancellable = Rc::downgrade(&transferable_timer_rc);

        wheel[slot_index].push(timeoutable);
        self.timer_by_reference_number
            .insert(reference_number, cancellable);
    }

    fn transfer_later_to_soon(&mut self, soon_max_monotonic_time_milliseconds: Milliseconds) {
        while let Some(check_timer_rc) = self.later.peek() {
            if check_timer_rc.monotonic_time_milliseconds <= soon_max_monotonic_time_milliseconds {
                let transferable_timer_rc = self.later.pop().unwrap();

                self.transfer(transferable_timer_rc, WheelName::Soon);
            } else {
                break;
            }
        }
    }

    fn transfer_long_term_to_later(&mut self, later_max_monotonic_time_milliseconds: Milliseconds) {
        while let Some(check_timer_rc) = self.long_term.peek() {
            if check_timer_rc.monotonic_time_milliseconds <= later_max_monotonic_time_milliseconds {
                let transferable_timer_rc = self.long_term.pop().unwrap();

                self.transfer(transferable_timer_rc, WheelName::Later);
            } else {
                break;
            }
        }
    }
}

impl Default for Hierarchy {
    fn default() -> Hierarchy {
        let monotonic_time_milliseconds = monotonic::time_in_milliseconds();

        let soon_slot_index = ((monotonic_time_milliseconds % Self::SOON_TOTAL_MILLISECONDS)
            / Self::SOON_MILLISECONDS_PER_SLOT) as u16;
        // round down to nearest multiple
        let soon_slot_monotonic_time_milliseconds = (monotonic_time_milliseconds
            / Self::SOON_TOTAL_MILLISECONDS)
            * Self::SOON_TOTAL_MILLISECONDS;
        let soon = Wheel::new(
            Self::SOON_MILLISECONDS_PER_SLOT,
            soon_slot_index,
            soon_slot_monotonic_time_milliseconds,
        );

        // > The later wheel contain timers that are further away from 'pos'
        // > than the width of the soon timer wheel.
        // -- https://github.com/erlang/otp/blob/759ec896d7f254db2996cbb503c1ef883e6714b0/erts/emulator/beam/time.c#L68-L69
        let later_monotonic_time_milliseconds =
            soon_slot_monotonic_time_milliseconds + Self::SOON_TOTAL_MILLISECONDS;
        let later_slot_index = ((later_monotonic_time_milliseconds
            % Self::LATER_TOTAL_MILLISECONDS)
            / Self::LATER_MILLISECONDS_PER_SLOT) as u16;
        // round down to nearest multiple
        let later_slot_monotonic_time_milliseconds = (later_monotonic_time_milliseconds
            / Self::LATER_MILLISECONDS_PER_SLOT)
            * Self::LATER_MILLISECONDS_PER_SLOT;
        let later = Wheel::new(
            Self::LATER_MILLISECONDS_PER_SLOT,
            later_slot_index,
            later_slot_monotonic_time_milliseconds,
        );

        Hierarchy {
            at_once: Default::default(),
            soon,
            later,
            long_term: Default::default(),
            timer_by_reference_number: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Position {
    AtOnce,
    Soon { slot_index: u16 },
    Later { slot_index: u16 },
    LongTerm,
}

#[cfg_attr(test, derive(Debug))]
struct Wheel {
    milliseconds_per_slot: Milliseconds,
    total_milliseconds: Milliseconds,
    slots: Vec<BinaryHeap<Rc<Timer>>>,
    slot_index: u16,
    slot_monotonic_time_milliseconds: Milliseconds,
}

impl Wheel {
    // same as values used in BEAM
    #[cfg(not(test))]
    const LENGTH: u16 = 1 << 14;
    // super short so that later and long term tests don't take forever
    #[cfg(test)]
    const LENGTH: u16 = 1 << 2;

    fn new(
        milliseconds_per_slot: Milliseconds,
        slot_index: u16,
        slot_monotonic_time_milliseconds: Milliseconds,
    ) -> Wheel {
        Wheel {
            milliseconds_per_slot,
            total_milliseconds: milliseconds_per_slot * (Self::LENGTH as Milliseconds),
            slots: vec![Default::default(); Self::LENGTH as usize],
            slot_index,
            slot_monotonic_time_milliseconds,
        }
    }

    fn max_monotonic_time_milliseconds(&self) -> Milliseconds {
        self.slot_monotonic_time_milliseconds + self.total_milliseconds - 1
    }

    fn next_slot(&mut self) {
        self.slot_index = (self.slot_index + 1) % Self::LENGTH;
        self.slot_monotonic_time_milliseconds += self.milliseconds_per_slot;
    }

    fn next_slot_monotonic_time_milliseconds(&self) -> Milliseconds {
        self.slot_monotonic_time_milliseconds + self.milliseconds_per_slot
    }

    fn peek(&self) -> Option<&Rc<Timer>> {
        self.slots[self.slot_index as usize].peek()
    }

    fn pop(&mut self) -> Option<Rc<Timer>> {
        self.slots[self.slot_index as usize].pop()
    }

    fn slot_index(&self, monotonic_time_milliseconds: Milliseconds) -> u16 {
        let milliseconds = monotonic_time_milliseconds - self.slot_monotonic_time_milliseconds;
        let slots = (milliseconds / self.milliseconds_per_slot) as u16;

        assert!(slots < Wheel::LENGTH, "monotonic_time_milliseconds ({:?}) is {:?} milliseconds ({:?} slots) away from slot_monotonic_time_milliseconds {:?}, but wheel only has {:?} slots ", monotonic_time_milliseconds, milliseconds, slots, self.slot_monotonic_time_milliseconds, Wheel::LENGTH);

        (self.slot_index + slots) % Wheel::LENGTH
    }
}

impl Index<u16> for Wheel {
    type Output = BinaryHeap<Rc<Timer>>;

    fn index(&self, slot_index: u16) -> &BinaryHeap<Rc<Timer>> {
        self.slots.index(slot_index as usize)
    }
}

impl IndexMut<u16> for Wheel {
    fn index_mut(&mut self, slot_index: u16) -> &mut BinaryHeap<Rc<Timer>> {
        self.slots.index_mut(slot_index as usize)
    }
}

enum WheelName {
    Soon,
    Later,
}

thread_local! {
   static HIERARCHY: RefCell<Hierarchy> = RefCell::new(Default::default());
}

#[cfg(test)]
pub fn at_once_milliseconds() -> Milliseconds {
    0
}

#[cfg(test)]
pub fn soon_milliseconds() -> Milliseconds {
    let milliseconds: Milliseconds = 1;

    assert!(milliseconds < Hierarchy::SOON_TOTAL_MILLISECONDS);

    milliseconds
}

#[cfg(test)]
pub fn later_milliseconds() -> Milliseconds {
    let milliseconds = Hierarchy::SOON_TOTAL_MILLISECONDS + 1;

    assert!(Hierarchy::SOON_TOTAL_MILLISECONDS < milliseconds);
    assert!(milliseconds < Hierarchy::LATER_TOTAL_MILLISECONDS);

    milliseconds
}

#[cfg(test)]
pub fn long_term_milliseconds() -> Milliseconds {
    let milliseconds = Hierarchy::SOON_TOTAL_MILLISECONDS + Hierarchy::LATER_TOTAL_MILLISECONDS + 1;

    assert!(Hierarchy::SOON_TOTAL_MILLISECONDS < milliseconds);
    assert!(
        (Hierarchy::SOON_TOTAL_MILLISECONDS + Hierarchy::LATER_TOTAL_MILLISECONDS) < milliseconds
    );

    milliseconds
}

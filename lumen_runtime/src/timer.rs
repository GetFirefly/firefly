use std::cmp::Ordering::{self, *};
use std::collections::HashMap;
use std::ops::{Index, IndexMut, RangeBounds};
use std::sync::{Arc, Mutex, Weak};
use std::vec::Drain;

use crate::atom::Existence::DoNotCare;
use crate::heap::{CloneIntoHeap, Heap};
use crate::message;
use crate::process::Process;
use crate::reference;
use crate::registry::{self, Registered};
use crate::scheduler::{self, Scheduler};
use crate::term::Term;
use crate::time::monotonic::{self, Milliseconds};

pub mod cancel;
pub mod read;
pub mod start;

pub fn cancel(timer_reference: &reference::local::Reference) -> Option<Milliseconds> {
    timer_reference.scheduler().and_then(|scheduler| {
        scheduler
            .hierarchy
            .write()
            .unwrap()
            .cancel(timer_reference.number())
    })
}

pub fn read(timer_reference: &reference::local::Reference) -> Option<Milliseconds> {
    timer_reference.scheduler().and_then(|scheduler| {
        scheduler
            .hierarchy
            .read()
            .unwrap()
            .read(timer_reference.number())
    })
}

pub fn start(
    monotonic_time_milliseconds: Milliseconds,
    destination: Destination,
    timeout: Timeout,
    process_message: Term,
    process: &Process,
) -> Term {
    let scheduler = Scheduler::current();
    let reference = scheduler.hierarchy.write().unwrap().start(
        monotonic_time_milliseconds,
        destination,
        timeout,
        process_message,
        process,
        &scheduler,
    );

    Term::box_reference(reference)
}

/// Times out the timers for the thread that have timed out since the last time `timeout` was
/// called.
pub fn timeout() {
    let scheduler = Scheduler::current();

    scheduler.hierarchy.write().unwrap().timeout(&scheduler.id);
}

#[derive(Clone, Debug)]
pub enum Destination {
    Name(Term),
    Process(Weak<Process>),
}

#[cfg_attr(test, derive(Debug))]
pub struct Hierarchy {
    at_once: Slot,
    soon: Wheel,
    later: Wheel,
    long_term: Slot,
    timer_by_reference_number: HashMap<reference::local::Number, Weak<Timer>>,
}

impl Hierarchy {
    const SOON_MILLISECONDS_PER_SLOT: Milliseconds = 1;
    const SOON_TOTAL_MILLISECONDS: Milliseconds =
        Self::SOON_MILLISECONDS_PER_SLOT * (Wheel::LENGTH as Milliseconds);
    const LATER_MILLISECONDS_PER_SLOT: Milliseconds = Self::SOON_TOTAL_MILLISECONDS / 2;
    const LATER_TOTAL_MILLISECONDS: Milliseconds =
        Self::LATER_MILLISECONDS_PER_SLOT * (Wheel::LENGTH as Milliseconds);

    fn cancel(&mut self, timer_reference_number: reference::local::Number) -> Option<Milliseconds> {
        self.timer_by_reference_number
            .remove(&timer_reference_number)
            .and_then(|weak_timer| weak_timer.upgrade())
            .map(|arc_timer| {
                use Position::*;

                match *arc_timer.position.lock().unwrap() {
                    // can't be found in O(1), mark as canceled for later cleanup
                    AtOnce => self.at_once.cancel(timer_reference_number),
                    Soon { slot_index } => self.soon.cancel(slot_index, timer_reference_number),
                    Later { slot_index } => self.later.cancel(slot_index, timer_reference_number),
                    LongTerm => self.long_term.cancel(timer_reference_number),
                };

                arc_timer.monotonic_time_milliseconds - monotonic::time_in_milliseconds()
            })
    }

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

    fn read(&self, timer_reference_number: reference::local::Number) -> Option<Milliseconds> {
        self.timer_by_reference_number
            .get(&timer_reference_number)
            .and_then(|weak_timer| weak_timer.upgrade())
            .map(|rc_timer| {
                rc_timer.monotonic_time_milliseconds - monotonic::time_in_milliseconds()
            })
    }

    fn start(
        &mut self,
        monotonic_time_milliseconds: Milliseconds,
        destination: Destination,
        timeout: Timeout,
        process_message: Term,
        process: &Process,
        scheduler: &Scheduler,
    ) -> &'static reference::local::Reference {
        let reference = scheduler.next_reference(process);
        let reference_number = reference.number();
        let heap: Heap = Default::default();
        let message = process_message.clone_into_heap(&heap);
        let position = self.position(monotonic_time_milliseconds);

        let timer = Timer {
            reference_number,
            monotonic_time_milliseconds,
            destination,
            timeout,
            message_heap: Mutex::new(message::Heap {
                heap,
                term: message,
            }),
            position: Mutex::new(position),
        };

        let arc_timer = Arc::new(timer);
        let timeoutable = Arc::clone(&arc_timer);
        let cancellable = Arc::downgrade(&arc_timer);

        match position {
            Position::AtOnce => self.at_once.start(timeoutable),
            Position::Soon { slot_index } => self.soon.start(slot_index, timeoutable),
            Position::Later { slot_index } => self.later.start(slot_index, timeoutable),
            Position::LongTerm => self.long_term.start(timeoutable),
        }

        self.timer_by_reference_number
            .insert(reference_number, cancellable);

        reference
    }

    pub fn timeout(&mut self, scheduler_id: &scheduler::ID) {
        self.timeout_at_once(scheduler_id);

        let monotonic_time_milliseconds = monotonic::time_in_milliseconds();
        let milliseconds = monotonic_time_milliseconds - self.soon.slot_monotonic_time_milliseconds;

        for _ in 0..milliseconds {
            self.timeout_soon_slot(&scheduler_id);

            assert!(self.soon.is_empty());
            self.soon.next_slot();

            let soon_max_monotonic_time_milliseconds = self.soon.max_monotonic_time_milliseconds();

            if self.later.slot_monotonic_time_milliseconds <= soon_max_monotonic_time_milliseconds {
                self.transfer_later_to_soon(soon_max_monotonic_time_milliseconds);

                let later_next_slot_monotonic_time_milliseconds =
                    self.later.next_slot_monotonic_time_milliseconds();

                if later_next_slot_monotonic_time_milliseconds
                    <= soon_max_monotonic_time_milliseconds
                {
                    assert!(self.later.is_empty());
                    self.later.next_slot();

                    let later_max_monotonic_time_milliseconds =
                        self.later.max_monotonic_time_milliseconds();

                    self.transfer_long_term_to_later(later_max_monotonic_time_milliseconds);
                }
            }
        }
    }

    fn timeout_at_once(&mut self, scheduler_id: &scheduler::ID) {
        for arc_timer in self.at_once.drain(..) {
            self.timer_by_reference_number
                .remove(&arc_timer.reference_number);

            Arc::try_unwrap(arc_timer).unwrap().timeout(scheduler_id)
        }
    }

    fn timeout_soon_slot(&mut self, scheduler_id: &scheduler::ID) {
        for arc_timer in self.soon.drain(..) {
            self.timer_by_reference_number
                .remove(&arc_timer.reference_number);

            Arc::try_unwrap(arc_timer).unwrap().timeout(scheduler_id)
        }
    }

    fn transfer(&mut self, mut transferable_arc_timer: Arc<Timer>, wheel_name: WheelName) {
        let wheel = match wheel_name {
            WheelName::Soon => &mut self.soon,
            WheelName::Later => &mut self.later,
        };

        let slot_index = wheel.slot_index(transferable_arc_timer.monotonic_time_milliseconds);

        let position = match wheel_name {
            WheelName::Soon => Position::Soon { slot_index },
            WheelName::Later => Position::Later { slot_index },
        };

        // Remove weak reference to allow `get_mut`.
        let reference_number = transferable_arc_timer.reference_number;
        self.timer_by_reference_number.remove(&reference_number);

        *Arc::get_mut(&mut transferable_arc_timer)
            .unwrap()
            .position
            .lock()
            .unwrap() = position;

        let timeoutable = Arc::clone(&transferable_arc_timer);
        let cancellable = Arc::downgrade(&transferable_arc_timer);

        wheel.start(slot_index, timeoutable);
        self.timer_by_reference_number
            .insert(reference_number, cancellable);
    }

    fn transfer_later_to_soon(&mut self, soon_max_monotonic_time_milliseconds: Milliseconds) {
        let transferable_arc_timers: Vec<Arc<Timer>> = self
            .later
            .drain_before_or_at(soon_max_monotonic_time_milliseconds)
            .collect();

        for arc_timer in transferable_arc_timers {
            self.transfer(arc_timer, WheelName::Soon)
        }
    }

    fn transfer_long_term_to_later(&mut self, later_max_monotonic_time_milliseconds: Milliseconds) {
        let transferable_arc_timers: Vec<Arc<Timer>> = self
            .long_term
            .drain_before_or_at(later_max_monotonic_time_milliseconds)
            .collect();

        for arc_timer in transferable_arc_timers {
            self.transfer(arc_timer, WheelName::Later);
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

#[derive(Debug)]
pub enum Timeout {
    // Sends only the `Timer` `message`
    Message,
    // Sends `{:timeout, timer_reference, message}`
    TimeoutTuple,
}

#[derive(Debug)]
struct Timer {
    // Can't be a `Boxed` `LocalReference` `Term` because those are boxed and the original Process
    // could GC the unboxed `LocalReference` `Term`.
    reference_number: reference::local::Number,
    monotonic_time_milliseconds: Milliseconds,
    destination: Destination,
    message_heap: Mutex<message::Heap>,
    timeout: Timeout,
    position: Mutex<Position>,
}

impl Timer {
    fn timeout(self, scheduler_id: &scheduler::ID) {
        match &self.destination {
            Destination::Name(ref name) => {
                let readable_registry = registry::RW_LOCK_REGISTERED_BY_NAME.read().unwrap();

                match readable_registry.get(name) {
                    Some(Registered::Process(destination_weak_process)) => {
                        match destination_weak_process.upgrade() {
                            Some(destination_arc_process) => {
                                let (heap, timeout_message) = self.timeout_message(scheduler_id);

                                destination_arc_process.send_heap_message(heap, timeout_message);
                            }
                            None => (),
                        }
                    }
                    None => (),
                }
            }
            Destination::Process(destination_process_weak) => {
                if let Some(destination_process_arc) = destination_process_weak.upgrade() {
                    let (heap, timeout_message) = self.timeout_message(scheduler_id);

                    destination_process_arc.send_heap_message(heap, timeout_message);
                }
            }
        }
    }

    fn timeout_message(self, scheduler_id: &scheduler::ID) -> (Heap, Term) {
        let message_heap = self.message_heap.into_inner().unwrap();

        let message = match self.timeout {
            Timeout::Message => message_heap.term,
            Timeout::TimeoutTuple => {
                let reference = message_heap
                    .heap
                    .local_reference(&scheduler_id, self.reference_number);
                let reference_term = Term::box_reference(reference);

                let tuple = message_heap.heap.slice_to_tuple(&[
                    Term::str_to_atom("timeout", DoNotCare).unwrap(),
                    reference_term,
                    message_heap.term,
                ]);

                Term::box_reference(tuple)
            }
        };

        (message_heap.heap, message)
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

#[derive(Clone, Copy, Debug)]
enum Position {
    AtOnce,
    Soon { slot_index: u16 },
    Later { slot_index: u16 },
    LongTerm,
}

/// A slot in the Hierarchy (for `at_once` and `long_term`) or a slot in a `Wheel` (for `soon` and
/// `later`).
#[cfg_attr(test, derive(Debug))]
#[derive(Clone, Default)]
struct Slot(Vec<Arc<Timer>>);

impl Slot {
    fn cancel(&mut self, reference_number: reference::local::Number) -> Option<Arc<Timer>> {
        self.0
            .iter()
            .position(|timer_rc| timer_rc.reference_number == reference_number)
            .map(|index| self.0.remove(index))
    }

    fn drain<R>(&mut self, range: R) -> Drain<Arc<Timer>>
    where
        R: RangeBounds<usize>,
    {
        self.0.drain(range)
    }

    fn drain_before_or_at(
        &mut self,
        max_monotonic_time_milliseconds: Milliseconds,
    ) -> Drain<Arc<Timer>> {
        let exclusive_end_bound = self
            .0
            .binary_search_by(|timer_rc| {
                match timer_rc
                    .monotonic_time_milliseconds
                    .cmp(&max_monotonic_time_milliseconds)
                {
                    Equal => Less,
                    ordering => ordering,
                }
            })
            .unwrap_err();

        self.0.drain(0..exclusive_end_bound)
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn start(&mut self, arc_timer: Arc<Timer>) {
        let index = self
            .0
            .binary_search_by_key(
                &(
                    arc_timer.monotonic_time_milliseconds,
                    arc_timer.reference_number,
                ),
                |existing_arc_timer| {
                    (
                        existing_arc_timer.monotonic_time_milliseconds,
                        existing_arc_timer.reference_number,
                    )
                },
            )
            .unwrap_err();

        self.0.insert(index, arc_timer)
    }
}

type SlotIndex = u16;

#[cfg_attr(test, derive(Debug))]
struct Wheel {
    milliseconds_per_slot: Milliseconds,
    total_milliseconds: Milliseconds,
    slots: Vec<Slot>,
    slot_index: u16,
    slot_monotonic_time_milliseconds: Milliseconds,
}

impl Wheel {
    // same as values used in BEAM
    #[cfg(not(test))]
    const LENGTH: SlotIndex = 1 << 14;
    // super short so that later and long term tests don't take forever
    #[cfg(test)]
    const LENGTH: SlotIndex = 1 << 2;

    fn new(
        milliseconds_per_slot: Milliseconds,
        slot_index: SlotIndex,
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

    fn cancel(
        &mut self,
        slot_index: SlotIndex,
        reference_number: reference::local::Number,
    ) -> Option<Arc<Timer>> {
        self.slots[slot_index as usize].cancel(reference_number)
    }

    fn drain<R>(&mut self, range: R) -> Drain<Arc<Timer>>
    where
        R: RangeBounds<usize>,
    {
        self.slots[self.slot_index as usize].drain(range)
    }

    fn drain_before_or_at(
        &mut self,
        max_monotonic_time_milliseconds: Milliseconds,
    ) -> Drain<Arc<Timer>> {
        self.slots[self.slot_index as usize].drain_before_or_at(max_monotonic_time_milliseconds)
    }

    fn is_empty(&self) -> bool {
        self.slots[self.slot_index as usize].is_empty()
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

    fn slot_index(&self, monotonic_time_milliseconds: Milliseconds) -> u16 {
        let milliseconds = monotonic_time_milliseconds - self.slot_monotonic_time_milliseconds;
        let slots = (milliseconds / self.milliseconds_per_slot) as u16;

        assert!(slots < Wheel::LENGTH, "monotonic_time_milliseconds ({:?}) is {:?} milliseconds ({:?} slots) away from slot_monotonic_time_milliseconds {:?}, but wheel only has {:?} slots ", monotonic_time_milliseconds, milliseconds, slots, self.slot_monotonic_time_milliseconds, Wheel::LENGTH);

        (self.slot_index + slots) % Wheel::LENGTH
    }

    fn start(&mut self, slot_index: SlotIndex, arc_timer: Arc<Timer>) {
        self.slots[slot_index as usize].start(arc_timer)
    }
}

impl Index<u16> for Wheel {
    type Output = Slot;

    fn index(&self, slot_index: u16) -> &Slot {
        self.slots.index(slot_index as usize)
    }
}

impl IndexMut<u16> for Wheel {
    fn index_mut(&mut self, slot_index: u16) -> &mut Slot {
        self.slots.index_mut(slot_index as usize)
    }
}

enum WheelName {
    Soon,
    Later,
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

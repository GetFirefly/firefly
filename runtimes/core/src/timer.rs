use core::cmp::Ordering::{self, *};
use core::fmt::{self, Debug};
use core::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, RangeBounds, Rem};
use core::ptr::NonNull;

use std::sync::{Arc, Weak};
use std::vec::Drain;

use hashbrown::HashMap;

use liblumen_core::locks::Mutex;

use liblumen_alloc::borrow::CloneToProcess;
use liblumen_alloc::erts::exception::AllocResult;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::Process;
use liblumen_alloc::time::{Milliseconds, Monotonic};

use crate::registry;
use crate::scheduler::{self, Scheduled, Scheduler};
use crate::time::monotonic;

pub fn cancel(timer_reference: &Reference) -> Option<Milliseconds> {
    timer_reference.scheduler().and_then(|scheduler| {
        scheduler
            .hierarchy()
            .write()
            .cancel(timer_reference.number())
    })
}

pub fn read(timer_reference: &Reference) -> Option<Milliseconds> {
    timer_reference
        .scheduler()
        .and_then(|scheduler| scheduler.hierarchy().read().read(timer_reference.number()))
}

pub fn start(
    monotonic: Monotonic,
    event: SourceEvent,
    arc_process: Arc<Process>,
) -> AllocResult<Term> {
    let arc_scheduler = scheduler::current();

    let result = arc_scheduler.hierarchy().write().start(
        monotonic,
        event,
        arc_process,
        arc_scheduler.clone(),
    );

    result
}

/// Times out the timers for the thread that have timed out since the last time `timeout` was
/// called.
pub fn timeout() {
    scheduler::current().hierarchy().write().timeout();
}

#[derive(Debug)]
pub struct Message {
    pub heap_fragment: NonNull<liblumen_alloc::erts::HeapFragment>,
    pub term: Term,
}

#[derive(Debug)]
pub struct HeapFragment {
    pub heap_fragment: NonNull<liblumen_alloc::erts::HeapFragment>,
    pub term: Term,
}

#[derive(Clone, Debug)]
pub enum Destination {
    Name(Atom),
    Process(Weak<Process>),
}

pub struct Hierarchy {
    at_once: Slot,
    soon: Wheel,
    later: Wheel,
    long_term: Slot,
    timer_by_reference_number: HashMap<ReferenceNumber, Weak<Timer>>,
}
impl Hierarchy {
    const SOON_MILLISECONDS_PER_SLOT: MillisecondsPerSlot = MillisecondsPerSlot(1);
    const SOON_TOTAL_MILLISECONDS: Milliseconds =
        Self::SOON_MILLISECONDS_PER_SLOT.const_mul(Wheel::SLOTS);
    const LATER_MILLISECONDS_PER_SLOT: MillisecondsPerSlot =
        MillisecondsPerSlot(Self::SOON_TOTAL_MILLISECONDS.const_div(2).as_u64());
    const LATER_TOTAL_MILLISECONDS: Milliseconds =
        Self::LATER_MILLISECONDS_PER_SLOT.const_mul(Wheel::SLOTS);

    pub fn cancel(&mut self, timer_reference_number: ReferenceNumber) -> Option<Milliseconds> {
        self.timer_by_reference_number
            .remove(&timer_reference_number)
            .and_then(|weak_timer| weak_timer.upgrade())
            .map(|arc_timer| {
                use Position::*;

                match *arc_timer.position.lock() {
                    // can't be found in O(1), mark as canceled for later cleanup
                    AtOnce => self.at_once.cancel(timer_reference_number),
                    Soon { slot_index } => self.soon.cancel(slot_index, timer_reference_number),
                    Later { slot_index } => self.later.cancel(slot_index, timer_reference_number),
                    LongTerm => self.long_term.cancel(timer_reference_number),
                };

                arc_timer.milliseconds_remaining()
            })
    }

    fn position(&self, monotonic: Monotonic) -> Position {
        if monotonic < self.soon.slot_monotonic {
            Position::AtOnce
        } else if monotonic < self.later.slot_monotonic {
            Position::Soon {
                slot_index: self.soon.slot_index(monotonic),
            }
        } else if monotonic < (self.later.slot_monotonic + Self::LATER_TOTAL_MILLISECONDS) {
            Position::Later {
                slot_index: self.later.slot_index(monotonic),
            }
        } else {
            Position::LongTerm
        }
    }

    pub fn read(&self, timer_reference_number: ReferenceNumber) -> Option<Milliseconds> {
        self.timer_by_reference_number
            .get(&timer_reference_number)
            .and_then(|weak_timer| weak_timer.upgrade())
            .map(|rc_timer| rc_timer.milliseconds_remaining())
    }

    pub fn start(
        &mut self,
        monotonic: Monotonic,
        source_event: SourceEvent,
        arc_process: Arc<Process>,
        arc_scheduler: Arc<dyn Scheduler>,
    ) -> AllocResult<Term> {
        let reference_number = arc_scheduler.next_reference_number();
        let process_reference =
            arc_process.reference_from_scheduler(arc_scheduler.id(), reference_number);

        let destination_event = match source_event {
            SourceEvent::Message {
                destination,
                format,
                term,
            } => {
                let (heap_fragment_message, heap_fragment) = match format {
                    Format::Message => term.clone_to_fragment()?,
                    Format::TimeoutTuple => {
                        let tag = Atom::str_to_term("timeout");
                        let process_tuple =
                            arc_process.tuple_from_slice(&[tag, process_reference, term]);

                        process_tuple.clone_to_fragment()?
                    }
                };
                let heap_fragment = Mutex::new(HeapFragment {
                    heap_fragment,
                    term: heap_fragment_message,
                });
                DestinationEvent::Message {
                    destination,
                    heap_fragment,
                }
            }
            SourceEvent::StopWaiting => DestinationEvent::StopWaiting {
                process: Arc::downgrade(&arc_process),
            },
        };

        let position = self.position(monotonic);

        let timer = Timer {
            reference_number,
            monotonic,
            event: destination_event,
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

        Ok(process_reference)
    }

    pub fn timeout(&mut self) {
        self.timeout_at_once();

        let monotonic = monotonic::time();
        let milliseconds = monotonic - self.soon.slot_monotonic;

        for _ in 0..milliseconds.into() {
            self.timeout_soon_slot();

            assert!(self.soon.is_empty());
            self.soon.next_slot();

            let soon_max_monotonic = self.soon.max_monotonic();

            if self.later.slot_monotonic <= soon_max_monotonic {
                self.transfer_later_to_soon(soon_max_monotonic);

                let later_next_slot_monotonic = self.later.next_slot_monotonic();

                if later_next_slot_monotonic <= soon_max_monotonic {
                    assert!(self.later.is_empty());
                    self.later.next_slot();

                    let later_max_monotonic = self.later.max_monotonic();

                    self.transfer_long_term_to_later(later_max_monotonic);
                }
            }
        }
    }

    fn timeout_at_once(&mut self) {
        for arc_timer in self.at_once.drain(..) {
            self.timer_by_reference_number
                .remove(&arc_timer.reference_number);

            Self::timeout_arc_timer(arc_timer);
        }
    }

    fn timeout_soon_slot(&mut self) {
        for arc_timer in self.soon.drain(..) {
            self.timer_by_reference_number
                .remove(&arc_timer.reference_number);

            Self::timeout_arc_timer(arc_timer);
        }
    }

    fn timeout_arc_timer(arc_timer: Arc<Timer>) {
        match Arc::try_unwrap(arc_timer) {
            Ok(timer) => timer.timeout(),
            Err(_) => panic!("Timer Dropped"),
        }
    }

    fn transfer(&mut self, mut transferable_arc_timer: Arc<Timer>, wheel_name: WheelName) {
        let wheel = match wheel_name {
            WheelName::Soon => &mut self.soon,
            WheelName::Later => &mut self.later,
        };

        let slot_index = wheel.slot_index(transferable_arc_timer.monotonic);

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
            .lock() = position;

        let timeoutable = Arc::clone(&transferable_arc_timer);
        let cancellable = Arc::downgrade(&transferable_arc_timer);

        wheel.start(slot_index, timeoutable);
        self.timer_by_reference_number
            .insert(reference_number, cancellable);
    }

    fn transfer_later_to_soon(&mut self, soon_max_monotonic: Monotonic) {
        let transferable_arc_timers: Vec<Arc<Timer>> =
            self.later.drain_before_or_at(soon_max_monotonic).collect();

        for arc_timer in transferable_arc_timers {
            self.transfer(arc_timer, WheelName::Soon)
        }
    }

    fn transfer_long_term_to_later(&mut self, later_max_monotonic: Monotonic) {
        let transferable_arc_timers: Vec<Arc<Timer>> = self
            .long_term
            .drain_before_or_at(later_max_monotonic)
            .collect();

        for arc_timer in transferable_arc_timers {
            self.transfer(arc_timer, WheelName::Later);
        }
    }
}

impl Debug for Hierarchy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("Timers\n")?;
        write!(f, "At Once:\n{:?}\n", self.at_once)?;
        write!(f, "Soon:\n{:?}", self.soon)?;
        write!(f, "Later:\n{:?}", self.later)?;
        write!(f, "Long Term:\n{:?}\n", self.long_term)
    }
}

impl Default for Hierarchy {
    fn default() -> Hierarchy {
        let monotonic = monotonic::time();

        let soon_slot_index = SlotIndex::from_monotonic(
            monotonic,
            Self::SOON_TOTAL_MILLISECONDS,
            Self::SOON_MILLISECONDS_PER_SLOT,
        );
        // round down to nearest multiple
        let soon_slot_monotonic = monotonic.round_down(Self::SOON_TOTAL_MILLISECONDS.into());
        let soon = Wheel::new(
            Self::SOON_MILLISECONDS_PER_SLOT,
            soon_slot_index,
            soon_slot_monotonic,
        );

        // > The later wheel contain timers that are further away from 'pos'
        // > than the width of the soon timer wheel.
        // -- https://github.com/erlang/otp/blob/759ec896d7f254db2996cbb503c1ef883e6714b0/erts/emulator/beam/time.c#L68-L69
        let later_monotonic = soon_slot_monotonic + Self::SOON_TOTAL_MILLISECONDS;
        let later_slot_index = SlotIndex::from_monotonic(
            later_monotonic,
            Self::LATER_TOTAL_MILLISECONDS,
            Self::LATER_MILLISECONDS_PER_SLOT,
        );
        // round down to nearest multiple
        let later_slot_monotonic =
            later_monotonic.round_down(Self::LATER_MILLISECONDS_PER_SLOT.into());
        let later = Wheel::new(
            Self::LATER_MILLISECONDS_PER_SLOT,
            later_slot_index,
            later_slot_monotonic,
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

// Hierarchies belong to Schedulers and Schedulers will never change threads
unsafe impl Send for Hierarchy {}
unsafe impl Sync for Hierarchy {}

#[derive(Clone, Copy)]
pub struct MillisecondsPerSlot(u64);

impl MillisecondsPerSlot {
    const fn const_mul(self, slots: Slots) -> Milliseconds {
        Milliseconds(self.0 * (slots.0 as u64))
    }
}

impl Add<MillisecondsPerSlot> for Monotonic {
    type Output = Monotonic;

    fn add(self, rhs: MillisecondsPerSlot) -> Monotonic {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign<MillisecondsPerSlot> for Monotonic {
    fn add_assign(&mut self, rhs: MillisecondsPerSlot) {
        self.0 += rhs.0
    }
}

impl Div<MillisecondsPerSlot> for Milliseconds {
    type Output = Slots;

    fn div(self, rhs: MillisecondsPerSlot) -> Slots {
        Slots((self.0 / rhs.0) as u16)
    }
}

impl From<MillisecondsPerSlot> for u64 {
    fn from(milliseconds_per_slot: MillisecondsPerSlot) -> u64 {
        milliseconds_per_slot.0
    }
}

impl Mul<Slots> for MillisecondsPerSlot {
    type Output = Milliseconds;

    fn mul(self, slots: Slots) -> Milliseconds {
        self.const_mul(slots)
    }
}

/// Event coming from source
#[derive(Debug)]
pub enum SourceEvent {
    Message {
        destination: Destination,
        format: Format,
        term: Term,
    },
    StopWaiting,
}

/// Format of `SourceEvent` `Message`
#[derive(Debug)]
pub enum Format {
    /// Sends only the `Timer` `message`
    Message,
    /// Sends `{:timeout, timer_reference, message}`
    TimeoutTuple,
}

struct Timer {
    // Can't be a `Boxed` `LocalReference` `Term` because those are boxed and the original Process
    // could GC the unboxed `LocalReference` `Term`.
    reference_number: ReferenceNumber,
    monotonic: Monotonic,
    event: DestinationEvent,
    position: Mutex<Position>,
}

impl Timer {
    fn milliseconds_remaining(&self) -> Milliseconds {
        // The timer may be read when it is past its timeout, but it has not been timed-out
        // by the scheduler.  Without this, an underflow would occur.
        // `0` is returned on underflow because that is what Erlang returns.
        match self.monotonic.checked_sub(monotonic::time()) {
            Some(difference) => difference,
            None => Milliseconds(0),
        }
    }

    fn timeout(self) {
        match self.event {
            DestinationEvent::Message {
                destination,
                heap_fragment,
            } => {
                let option_destination_arc_process = match &destination {
                    Destination::Name(ref name) => registry::atom_to_process(name),
                    Destination::Process(destination_process_weak) => {
                        destination_process_weak.upgrade()
                    }
                };

                if let Some(destination_arc_process) = option_destination_arc_process {
                    let HeapFragment {
                        heap_fragment,
                        term,
                    } = heap_fragment.into_inner();

                    destination_arc_process.send_heap_message(heap_fragment, term);
                    destination_arc_process.stop_waiting();
                }
            }
            DestinationEvent::StopWaiting { process } => {
                if let Some(destination_arc_process) = process.upgrade() {
                    // `__lumen_builtin_receive_wait` will notice it has timed out, so only need to
                    // stop waiting

                    // change process status
                    destination_arc_process.stop_waiting();

                    // move to correct run_queue
                    destination_arc_process
                        .scheduler()
                        .unwrap()
                        .stop_waiting(&destination_arc_process);
                }
            }
        }
    }
}

impl Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "  {} ", self.monotonic)?;

        match &self.event {
            DestinationEvent::Message {
                destination,
                heap_fragment,
            } => {
                let HeapFragment { term, .. } = *heap_fragment.lock();
                write!(f, "{} -> ", term)?;

                match destination {
                    Destination::Process(weak_process) => fmt_weak_process(weak_process, f),
                    Destination::Name(name) => write!(f, "{}", name),
                }?;
            }
            DestinationEvent::StopWaiting { process } => {
                fmt_weak_process(process, f)?;
                write!(f, " stop waiting")?;
            }
        }

        if self.monotonic <= monotonic::time() {
            write!(f, " (expired)")?;
        }

        Ok(())
    }
}

fn fmt_weak_process(weak_process: &Weak<Process>, f: &mut fmt::Formatter) -> fmt::Result {
    match weak_process.upgrade() {
        Some(arc_process) => write!(f, "{}", arc_process),
        None => write!(f, "Dead Process"),
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
            .monotonic
            .cmp(&self.monotonic)
            .then_with(|| other.reference_number.cmp(&self.reference_number))
    }
}

impl PartialOrd for Timer {
    fn partial_cmp(&self, other: &Timer) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Event sent to destination
enum DestinationEvent {
    /// Send message in `heap_fragment` to `destination`.
    Message {
        destination: Destination,
        heap_fragment: Mutex<HeapFragment>,
    },
    /// Stop `process` from waiting
    StopWaiting { process: Weak<Process> },
}

#[derive(Clone, Copy)]
#[cfg_attr(debug_assertions, derive(Debug))]
enum Position {
    AtOnce,
    Soon { slot_index: SlotIndex },
    Later { slot_index: SlotIndex },
    LongTerm,
}

/// A slot in the Hierarchy (for `at_once` and `long_term`) or a slot in a `Wheel` (for `soon` and
/// `later`).
#[derive(Clone, Default)]
struct Slot(Vec<Arc<Timer>>);

impl Slot {
    fn cancel(&mut self, reference_number: ReferenceNumber) -> Option<Arc<Timer>> {
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

    fn drain_before_or_at(&mut self, max_monotonic: Monotonic) -> Drain<Arc<Timer>> {
        let exclusive_end_bound = self
            .0
            .binary_search_by(|arc_timer| match arc_timer.monotonic.cmp(&max_monotonic) {
                Equal => Less,
                ordering => ordering,
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
                &(arc_timer.monotonic, arc_timer.reference_number),
                |existing_arc_timer| {
                    (
                        existing_arc_timer.monotonic,
                        existing_arc_timer.reference_number,
                    )
                },
            )
            .unwrap_err();

        self.0.insert(index, arc_timer)
    }
}

impl Debug for Slot {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0.is_empty() {
            write!(f, "  No timers\n")?;
        } else {
            for arc_timer in &self.0 {
                write!(f, "  {:?}\n", arc_timer)?;
            }
        }

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
struct SlotIndex(u16);

impl SlotIndex {
    fn from_monotonic(
        monotonic: Monotonic,
        total: Milliseconds,
        milliseconds_per_slot: MillisecondsPerSlot,
    ) -> SlotIndex {
        let remaining = monotonic % total;
        SlotIndex((remaining.0 / milliseconds_per_slot.0) as u16)
    }
}

impl Add<u16> for SlotIndex {
    type Output = SlotIndex;

    fn add(self, rhs: u16) -> SlotIndex {
        Self(self.0 + rhs)
    }
}

impl Add<Slots> for SlotIndex {
    type Output = SlotIndex;

    fn add(self, rhs: Slots) -> SlotIndex {
        Self(self.0 + rhs.0)
    }
}

impl Rem<Slots> for SlotIndex {
    type Output = SlotIndex;

    fn rem(self, rhs: Slots) -> SlotIndex {
        Self(self.0 % rhs.0)
    }
}

#[derive(Debug, Eq, PartialEq, PartialOrd)]
pub struct Slots(u16);

struct Wheel {
    milliseconds_per_slot: MillisecondsPerSlot,
    total_milliseconds: Milliseconds,
    slots: Vec<Slot>,
    slot_index: SlotIndex,
    slot_monotonic: Monotonic,
}

impl Wheel {
    // same as values used in BEAM
    const SLOTS: Slots = Slots(1 << 14);

    fn new(
        milliseconds_per_slot: MillisecondsPerSlot,
        slot_index: SlotIndex,
        slot_monotonic: Monotonic,
    ) -> Wheel {
        Wheel {
            milliseconds_per_slot,
            total_milliseconds: milliseconds_per_slot * Self::SLOTS,
            slots: vec![Default::default(); Self::SLOTS.0 as usize],
            slot_index,
            slot_monotonic,
        }
    }

    fn cancel(
        &mut self,
        slot_index: SlotIndex,
        reference_number: ReferenceNumber,
    ) -> Option<Arc<Timer>> {
        self.slots[slot_index.0 as usize].cancel(reference_number)
    }

    fn drain<R>(&mut self, range: R) -> Drain<Arc<Timer>>
    where
        R: RangeBounds<usize>,
    {
        self.slots[self.slot_index.0 as usize].drain(range)
    }

    fn drain_before_or_at(&mut self, max_monotonic: Monotonic) -> Drain<Arc<Timer>> {
        self.slots[self.slot_index.0 as usize].drain_before_or_at(max_monotonic)
    }

    fn is_empty(&self) -> bool {
        self.slots[self.slot_index.0 as usize].is_empty()
    }

    fn max_monotonic(&self) -> Monotonic {
        self.slot_monotonic + self.total_milliseconds - Milliseconds(1)
    }

    fn next_slot(&mut self) {
        self.slot_index = (self.slot_index + 1) % Self::SLOTS;
        self.slot_monotonic += self.milliseconds_per_slot;
    }

    fn next_slot_monotonic(&self) -> Monotonic {
        self.slot_monotonic + self.milliseconds_per_slot
    }

    fn slot_index(&self, monotonic: Monotonic) -> SlotIndex {
        let milliseconds = monotonic - self.slot_monotonic;
        let slots = milliseconds / self.milliseconds_per_slot;

        assert!(slots < Wheel::SLOTS, "monotonic ({:?}) is {:?} milliseconds ({:?} slots) away from slot_monotonic {:?}, but wheel only has {:?} slots ", monotonic, milliseconds, slots, self.slot_monotonic, Wheel::SLOTS);

        (self.slot_index + slots) % Wheel::SLOTS
    }

    fn start(&mut self, slot_index: SlotIndex, arc_timer: Arc<Timer>) {
        self.slots[slot_index.0 as usize].start(arc_timer)
    }
}

impl Debug for Wheel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(
            f,
            "  milliseconds per slot: {:?}",
            self.milliseconds_per_slot.0
        )?;
        writeln!(f, "  slots: {:?}", self.slots.len())?;
        writeln!(f, "  ____________")?;
        writeln!(f, "  total milliseconds: {:?}", self.total_milliseconds)?;
        writeln!(f, "")?;
        writeln!(f, "  slot index: {}", self.slot_index.0)?;

        write!(f, "  slot time: {} ms", self.slot_monotonic)?;

        if self.slot_monotonic <= monotonic::time() {
            writeln!(f, " (expired)")?;
        }

        let mut has_timers = false;

        for slot in &self.slots {
            for arc_timer in &slot.0 {
                has_timers = true;
                writeln!(f, "{:?}", arc_timer)?;
            }
        }

        if !has_timers {
            writeln!(f, "  No timers")?;
        }

        Ok(())
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

pub fn at_once_milliseconds() -> Milliseconds {
    Milliseconds(0)
}

pub fn soon_milliseconds() -> Milliseconds {
    let milliseconds = Milliseconds(1);

    assert!(milliseconds < Hierarchy::SOON_TOTAL_MILLISECONDS);

    milliseconds
}

pub fn later_milliseconds() -> Milliseconds {
    let milliseconds = Hierarchy::SOON_TOTAL_MILLISECONDS + Milliseconds(1);

    assert!(Hierarchy::SOON_TOTAL_MILLISECONDS < milliseconds);
    assert!(milliseconds < Hierarchy::LATER_TOTAL_MILLISECONDS);

    milliseconds
}

pub fn long_term_milliseconds() -> Milliseconds {
    let milliseconds =
        Hierarchy::SOON_TOTAL_MILLISECONDS + Hierarchy::LATER_TOTAL_MILLISECONDS + Milliseconds(1);

    assert!(Hierarchy::SOON_TOTAL_MILLISECONDS < milliseconds);
    assert!(
        (Hierarchy::SOON_TOTAL_MILLISECONDS + Hierarchy::LATER_TOTAL_MILLISECONDS) < milliseconds
    );

    milliseconds
}

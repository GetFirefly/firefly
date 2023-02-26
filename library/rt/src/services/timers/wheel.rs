use alloc::boxed::Box;
use core::fmt;
use core::mem::MaybeUninit;
use core::ops::Deref;
use core::ptr::NonNull;

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink, UnsafeRef};

use rustc_hash::FxHasher;

use firefly_system::time::{Duration, Timeout};

use crate::term::ReferenceId;

use super::{Timer, TimerError};

type HashMap<K, V> = hashbrown::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;

/// Represents an entry in the timer wheel
///
/// This struct is public because the timer wheel returns a list of entries on each tick,
/// and the list is an intrusive list whose links are held in this structure. For consumers
/// outside this module, the only meaningful thing to do with an entry is to unwrap it and
/// take out the timer inside.
pub struct TimerEntry {
    link: LinkedListLink,
    expiration: Expiration,
    timer: Timer,
}
impl TimerEntry {
    pub fn new(expiration: Expiration, timer: Timer) -> Self {
        Self {
            link: LinkedListLink::new(),
            expiration,
            timer,
        }
    }

    #[inline]
    pub fn id(&self) -> ReferenceId {
        self.timer.id()
    }

    /// Unwraps this entry and returns the timer it holds
    pub fn into_timer(self) -> Timer {
        self.timer
    }
}
impl fmt::Debug for TimerEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("TimerEntry")
            .field("expiration", &self.expiration)
            .field("timer", &self.timer)
            .finish()
    }
}

/// An `Expiration` value represents multiple things:
///
/// * What wheel does a timer belong to, e.g. short-term, long-term, or does it belong to overflow
/// * What slot in that wheel owns the timer
/// * What wheel and slot will the timer move to on the next tick which sees it
/// * The absolute time of expiry (relative to the timer wheel clock)
///
/// This type is used to help enforce a few invariants of the timer wheel system, as described above.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[repr(u32)]
pub enum Expiration {
    /// This timer is in the lowest level of the hierarchy, and will expire soon
    Soon(Time),
    /// This timer is in the second level of the hierarchy, representing a short-term expiration
    Short(Time),
    /// This timer is in the third level of the hierarchy, representing a medium-term expiration
    Medium(Time),
    /// This timer is in the fourth level of the hierarchy, representing a long-term expiration
    Long(Time),
    /// This timer has a duration beyond the maximum duration supported by the timer wheel
    ///
    /// It will be rescheduled to a lower level at some point in the future
    Overflow,
}
impl Expiration {
    /// This function is called when the tick of a [`TimerWheel`] is performed, and we are determining
    /// the course of action for all entries in the current slot. Namely, whether they should be
    /// considered expired, or if they have further levels of the timer wheel to travel through before
    /// expiry occurs.
    ///
    /// If this function returns `None`, it indicates that the corresponding entry should be considered expired.
    ///
    /// Otherwise, this function returns a new [`Expiration`] which reflects the next level of the hierarchy to
    /// place the entry in.
    ///
    /// NOTE: If this expiration is of type `Overflow`, it always returns the expiration unmodified, as this
    /// function is not intended for use in managing the overflow list.
    pub fn tick(self) -> Option<Self> {
        // In each of the following branches where a promotion to the next lowest level would occur;
        // if the slot for the next wheel is zero, that wheel is skipped, and promotion moves to the next
        // lowest level after that. If all remaining wheels are skipped, then expiry has occurred.
        match self {
            Self::Soon(_) => None,
            Self::Short(time) if time.soon() == 0 => None,
            Self::Short(time) => Some(Self::Soon(time)),
            Self::Medium(time) => match (time.short(), time.soon()) {
                (0, 0) => None,
                (0, _) => Some(Self::Soon(time)),
                (_, _) => Some(Self::Short(time)),
            },
            Self::Long(time) => match (time.medium(), time.short(), time.soon()) {
                (0, 0, 0) => None,
                (0, 0, _) => Some(Self::Soon(time)),
                (0, _, _) => Some(Self::Short(time)),
                (_, _, _) => Some(Self::Medium(time)),
            },
            Self::Overflow => Some(Self::Overflow),
        }
    }
}

/// Represents the current time in the hierarchical timer wheel
///
/// The "time" of the timer wheel is actually a clever combination of the
/// current slot indices of each wheel encoded as a byte of a big-endian u32,
/// in descending order of duration. This correspondance provides a means for
/// performing math on the time to obtain the slot indices at some future or
/// past point. Doing this on a raw u32 would be quite error prone, so this
/// type provides a safe abstraction for those operations, and for accessing
/// the slots by level in the hierarchy.
///
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Time([u8; 4]);
impl Default for Time {
    fn default() -> Self {
        Self([0, 0, 0, 0])
    }
}
impl Time {
    /// Constructs a new time given the current position in each of the four wheels
    #[inline]
    fn new(soon: u8, short: u8, medium: u8, long: u8) -> Self {
        Self([long, medium, short, soon])
    }

    /// Gets a raw representation of this time as a u32
    #[inline]
    fn as_u32(self) -> u32 {
        u32::from_be_bytes(self.0)
    }

    /// Returns the position in the lowest level timer wheel to which this time corresponds
    #[inline(always)]
    pub fn soon(&self) -> u8 {
        self.0[3]
    }

    /// Returns the position in the second level timer wheel to which this time corresponds
    #[inline(always)]
    pub fn short(&self) -> u8 {
        self.0[2]
    }

    /// Returns the position in the third level timer wheel to which this time corresponds
    #[inline(always)]
    pub fn medium(&self) -> u8 {
        self.0[1]
    }

    /// Returns the position in the fourth level timer wheel to which this time corresponds
    #[inline(always)]
    pub fn long(&self) -> u8 {
        self.0[0]
    }

    /// Derives an [`Expiration`] from this time after a desired duration
    ///
    /// Returns `Err` if expiration would occur immediately
    pub fn to_expiration(self, timeout: Timeout) -> Result<Expiration, ()> {
        let timeout = timeout.as_duration();

        if timeout > HierarchicalTimerWheel::MAX_SCHEDULE_DURATION {
            return Ok(Expiration::Overflow);
        }

        let absolute_time = self + timeout;
        let expiry_time = absolute_time ^ self;

        match expiry_time.0 {
            [0, 0, 0, 0] => Err(()),
            [0, 0, 0, _] => Ok(Expiration::Soon(absolute_time)),
            [0, 0, _, _] => Ok(Expiration::Short(absolute_time)),
            [0, _, _, _] => Ok(Expiration::Medium(absolute_time)),
            [_, _, _, _] => Ok(Expiration::Long(absolute_time)),
        }
    }

    #[inline]
    fn wrapping_add(self, rhs: u32) -> Self {
        Self(u32::to_be_bytes(self.as_u32().wrapping_add(rhs)))
    }
}
impl core::ops::Add<Duration> for Time {
    type Output = Time;

    fn add(self, duration: Duration) -> Self::Output {
        // Convert timeout to number of milliseconds in u32
        let duration = {
            let s = (duration.as_secs() * 1_000) as u32;
            let ms = duration.subsec_millis();
            s + ms
        };

        self.wrapping_add(duration)
    }
}
impl core::ops::BitXor<Time> for Time {
    type Output = Time;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(u32::to_be_bytes(self.as_u32() ^ rhs.as_u32()))
    }
}

intrusive_adapter!(pub TimerAdapter = UnsafeRef<TimerEntry>: TimerEntry { link: LinkedListLink });

/// A friendly type alias for the list type used by the wheel
pub type TimerList = LinkedList<TimerAdapter>;

/// A simple timer wheel which manages a single level of the hierarchy
///
/// Each wheel of this type represents a range of time divided up into 256 units,
/// which defines the "cycle time" and "tick time", respectively, of the wheel in
/// milliseconds. The cycle time is also the maximum duration representable by the
/// wheel.
///
/// For example, in the lowest level wheel, the cycle time is 256 milliseconds, and
/// the tick time is equivalent to 1 millisecond. For the second level however, the
/// cycle time is 65,536 milliseconds, and the tick time is 256 milliseconds - and so on.
///
/// The clock of this wheel is the slot index which it most recently visited, and each
/// tick increments it by 1, wrapping around to 0 on overflow.
pub struct TimerWheel {
    slots: [TimerList; 256],
    count: usize,
    current: u8,
}
impl TimerWheel {
    /// Create a new, empty wheel
    pub fn new() -> Self {
        let mut slots = MaybeUninit::<TimerList>::uninit_array::<256>();
        for slot in &mut slots {
            slot.write(TimerList::default());
        }

        Self {
            slots: unsafe { MaybeUninit::array_assume_init(slots) },
            count: 0,
            current: 0,
        }
    }

    /// Return the current position in the wheel, the "clock"
    #[inline]
    pub fn current(&self) -> u8 {
        self.current
    }

    /// Insert a new timer in the wheel in `slot`
    pub fn insert(&mut self, slot: u8, timer: UnsafeRef<TimerEntry>) {
        self.slots[slot as usize].push_back(timer);
        self.count += 1;
    }

    /// Advance the wheel one tick
    ///
    /// This returns a tuple containing the entries which expired this tick, and a boolean
    /// indicating whether or not this wheel has completed a cycle.
    pub fn tick(&mut self) -> (TimerList, bool) {
        let current = self.current;
        self.current = self.current.wrapping_add(1);
        let list = self.slots[current as usize].take();
        self.count -= list.iter().count();
        (list, self.current == 0)
    }

    /// This function can be used to fast-forward the clock to the next tick at which there are timers to process
    ///
    /// This is intended for use in testing and other scenarios where there is no need to advance the clock with real time
    ///
    /// This function returns true if a cycle was completed in the process of advancing the wheel.
    #[cfg(test)]
    #[allow(unused)]
    pub fn skip(&mut self) -> bool {
        match self.skippable() {
            // If this wheel is empty, we treat it as if we've completed a full cycle
            Skip::None => false,
            Skip::Some(n) => {
                // Converting `n` to u8 should always succeed here
                self.current = self.current.wrapping_add(n.try_into().unwrap());
                false
            }
            Skip::Empty | Skip::All => {
                // When we skip, it is to the end of the cycle, not to the beginning of a new cycle
                self.current = 255;
                true
            }
        }
    }

    /// Advances the wheel to the given slot without handling any timers that may have been present in earlier slots.
    ///
    /// This is intended for use by higher-level functions only.
    unsafe fn advance(&mut self, n: u8) {
        self.current = n;
    }

    fn skippable(&self) -> Skip {
        if self.count == 0 {
            return Skip::Empty;
        }

        let mut current = self.current as usize;
        let mut skippable = 0;
        loop {
            if self.slots[current].is_empty() {
                current = current.wrapping_add(1);
                if current == 0 {
                    return Skip::All;
                }
                skippable += 1;
            } else {
                // We've encountered a non-empty slot
                break;
            }
        }

        if skippable == 0 {
            Skip::None
        } else {
            Skip::Some(skippable)
        }
    }
}

/// A complete hierarchical timer wheel implementation
///
/// This builds on top of [`TimerWheel`] to provide a timer wheel capable of representing timers
/// with timeouts up to `Duration::MAX - 1` milliseconds into the future.
///
/// In addition, this wheel supports cancellation using the unique reference id associated
/// with each timer.
///
/// The hierarchy of this timer wheel consists of 4 sub-wheels, each corresponding to a different
/// range of time up to `u32::MAX` milliseconds, and an overflow list, which manages all of the
/// timers with timeouts greater than `u32::MAX` milliseconds. Each tick of the wheel may tick one
/// or more of the sub-wheels, starting with the lowest level wheel first, and working upwards for
/// each wheel that completes a cycle during the same tick. When the highest level wheel completes
/// a cycle, that corresponds to completion of a single cycle of the overall timer wheel. When such
/// a cycle is completed, the overflow list is visited, and entries are either promoted to a specific
/// wheel, or the timeout is recalculated and the entry is placed back in overflow.
///
/// When an entry in a wheel is visited during a tick, it is considered expired if it has no further
/// wheels to move through. Otherwise, it is moved to a lower level wheel in the hierarchy, based on
/// its expiry time.
///
/// NOTE: This timer wheel must be driven by some external process - it does not track real time automatically for you.
pub struct HierarchicalTimerWheel {
    wheels: [TimerWheel; 4],
    timers: HashMap<ReferenceId, UnsafeRef<TimerEntry>>,
    overflow: TimerList,
}
impl HierarchicalTimerWheel {
    /// The index of the lowest level wheel in the hierarchy
    const SOON: usize = 0;
    /// The index of the short-term wheel in the hierarchy
    const SHORT: usize = 1;
    /// The index of the medium-term wheel in the hierarchy
    const MEDIUM: usize = 2;
    /// The index of the long-term wheel in the hierarchy
    const LONG: usize = 3;
    /// The elapsed time (in milliseconds), per tick of the lowest level wheel
    const MS_PER_TICK_SOON: u32 = 1;
    /// The elapsed time (in milliseconds), per tick of the short-term wheel
    const MS_PER_TICK_SHORT: u32 = 1 << 8;
    /// The elapsed time (in milliseconds), per tick of the medium-term wheel
    const MS_PER_TICK_MEDIUM: u32 = 1 << 16;
    /// The elapsed time (in milliseconds), per tick of the long-term wheel
    const MS_PER_TICK_LONG: u32 = 1 << 24;
    /// The cycle time of the lowest level wheel
    ///     const SOON_LENGTH: u32 = 1 << 8;
    ///
    /// The cycle time of the short-term wheel
    ///     const SHORT_LENGTH: u32 = 1 << 16;
    ///
    /// The cycle time of the medium-term wheel
    ///     const MEDIUM_LENGTH: u32 = 1 << 24;
    ///
    /// The total cycle time
    const CYCLE_LENGTH: u64 = 1 << 32;
    /// The largest timeout duration allowed to be scheduled
    ///
    /// Anything larger than this goes in the overflow list
    const MAX_SCHEDULE_DURATION: Duration = Duration::from_millis(u32::MAX as u64);

    /// Creates a new wheel
    ///
    /// A system should only ever need one of these, but there is no limit to the number
    /// that can be created at the same time.
    pub fn new() -> Self {
        let mut wheels = MaybeUninit::<TimerWheel>::uninit_array::<4>();
        for wheel in &mut wheels {
            wheel.write(TimerWheel::new());
        }

        Self {
            wheels: unsafe { MaybeUninit::array_assume_init(wheels) },
            timers: HashMap::default(),
            overflow: TimerList::default(),
        }
    }

    /// Returns true if there are no timers registered with this wheel
    pub fn is_empty(&self) -> bool {
        self.timers.is_empty()
    }

    /// Inserts `timer` in the wheel.
    ///
    /// This may fail for a few reasons:
    ///
    /// * The timeout of the timer is infinite, which is not allowed
    /// * The timer would immediately expire
    ///
    /// Once inserted, the timer will remain in the wheel until its timeout is
    /// reached, or `cancel` is called with its reference id.
    pub fn insert(&mut self, timer: Timer) -> Result<NonNull<TimerEntry>, TimerError> {
        let timeout = timer.timeout();

        // Infinite timeouts should never reach the timer wheel
        if timeout.is_infinite() {
            return Err(TimerError::InvalidTimeout(timer));
        }

        // If the timeout is too far in the future, adjust its timeout relative to the
        // time remaining until the current cycle is complete, and place it in the overflow list
        if timeout >= Self::MAX_SCHEDULE_DURATION {
            let result = self.insert_overflow(Box::new(TimerEntry {
                link: LinkedListLink::new(),
                expiration: Expiration::Overflow,
                timer,
            }));
            return Ok(result);
        }

        // Otherwise, calculate the expiration for this timer, and insert an entry in
        // the wheel and slot which correspond to the absolute time of expiry
        let current_time = self.current_time_in_cycle();
        let expiration = match current_time.to_expiration(timeout) {
            Err(_) => return Err(TimerError::Expired(timer)),
            Ok(expiration) => expiration,
        };

        let entry = Box::new(TimerEntry {
            link: LinkedListLink::new(),
            expiration,
            timer,
        });
        let result =
            unsafe { NonNull::new_unchecked(entry.as_ref() as *const _ as *mut TimerEntry) };
        self.insert_to_wheel(entry);

        Ok(result)
    }

    fn insert_overflow(&mut self, mut entry: Box<TimerEntry>) -> NonNull<TimerEntry> {
        let remaining = Duration::from_millis(self.remaining_time_in_cycle());
        *entry.timer.timeout_mut() -= remaining;
        let id = entry.timer.id();
        let entry = UnsafeRef::from_box(entry);
        self.overflow.push_back(entry.clone());
        let result =
            unsafe { NonNull::new_unchecked(entry.as_ref() as *const _ as *mut TimerEntry) };
        if !self.timers.contains_key(&id) {
            self.timers.insert(id, entry);
        }
        result
    }

    fn insert_to_wheel(&mut self, entry: Box<TimerEntry>) {
        let wheel;
        let slot;

        match entry.expiration {
            Expiration::Soon(time) => {
                wheel = &mut self.wheels[Self::SOON];
                slot = time.soon();
            }
            Expiration::Short(time) => {
                wheel = &mut self.wheels[Self::SHORT];
                slot = time.short();
            }
            Expiration::Medium(time) => {
                wheel = &mut self.wheels[Self::MEDIUM];
                slot = time.medium();
            }
            Expiration::Long(time) => {
                wheel = &mut self.wheels[Self::LONG];
                slot = time.long();
            }
            // This variant is technically never reached with how this function
            // is called currently, but we can handle it properly anyway.
            Expiration::Overflow => {
                self.insert_overflow(entry);
                return;
            }
        }

        let id = entry.id();
        let entry = UnsafeRef::from_box(entry);
        wheel.insert(slot, entry.clone());
        if !self.timers.contains_key(&id) {
            self.timers.insert(id, entry);
        }
    }

    /// Cancel the timer with the given `id`
    ///
    /// Returns `Ok` if the timer was cancelled, and `Err` if no such timer could be found.
    pub fn cancel(&mut self, id: ReferenceId) -> Result<(), ()> {
        // While removing the reference from the lookup table ensures that
        // the entry will simply be pruned when encountered during a tick;
        // we can actually remove the timer from the wheel at the same time
        // due to our use of intrusive lists for storing entries, ensuring
        // that we don't have to wait until expiry to deallocate the memory
        // associated with the timer.
        match self.timers.remove_entry(&id) {
            Some((_id, entry)) => {
                // The expiration of a timer determines what wheel it is currently in,
                // as well as what slot of that wheel it is a member of. We use these
                // two bits of information to obtain a cursor into the entry list holding
                // this entry and remove it, then drop the allocation by converting the
                // UnsafeRef back into a box.
                //
                // # SAFETY
                //
                // There are a two invariants that must be upheld for this to be safe:
                //
                // * The only two UnsafeRefs allowed to exist to the timer must be the
                // two that we have here: the one in the lookup table, and the one in
                // the entry list.
                //
                // * If for some reason a timer is placed in a different wheel and slot
                // than the one determined by the expiration, that is definitely a bug,
                // but it will also cause this to fail horribly.
                match self.expiration_to_wheel_and_slot(entry.expiration) {
                    None => {
                        // This entry is in the overflow list
                        unsafe {
                            let mut cursor = self.overflow.cursor_mut_from_ptr(entry.deref());
                            UnsafeRef::into_box(cursor.remove().unwrap());
                        }
                    }
                    Some((wheel, slot)) => {
                        let wheel = &mut self.wheels[wheel];
                        unsafe {
                            let mut cursor = wheel.slots[slot].cursor_mut_from_ptr(entry.deref());
                            UnsafeRef::into_box(cursor.remove().unwrap());
                            wheel.count -= 1;
                        }
                    }
                }

                Ok(())
            }
            None => Err(()),
        }
    }

    /// Advances the timer wheel one tick, where each tick is intended to represent one millisecond of real time.
    ///
    /// The actual real time that elapses between ticks may actually be more or less than one millisecond though,
    /// this is determined by the process driving the timer wheel.
    ///
    /// A list of expired timer entries is returned, which should be used by the caller to actually execute the
    /// events associated with those entries.
    ///
    /// Recurring timers are automatically rescheduled when they fire, so there is no need to manage that elsewhere.
    pub fn tick(&mut self) -> TimerList {
        let mut result = TimerList::default();
        let mut reschedule = TimerList::default();
        let mut wheel = Self::SOON;

        // This loop is here because each tick of the hierarchical wheel might involve ticking
        // multiple wheels of the hierarchy. When ticking an individual wheel, it might signal that
        // it has completed its cycle, looping back around to the first slot. When that happens, we
        // proceed to ticking the next highest wheel in the hierarchy as well. This escalation stops
        // when we either hit a wheel which has not yet completed a cycle, or there are no more wheels
        // to tick, and we have an opportunity to process entries in the overflow list.
        loop {
            let (mut entries, cycle_completed) = self.wheels[wheel].tick();

            // For each entry in the list returned by the wheel, we have to decide whether that entry
            // has either: expired and should be returned in the result list; been cancelled and should
            // be pruned; or should moved to a lower level wheel as it has not yet expired.
            let mut cursor = entries.cursor_mut();
            while let Some(entry) = cursor.get() {
                // If the timer was cancelled, prune the entry
                if self.is_cancelled(entry) {
                    // The entry was removed from the timers map, so this is the last outstanding reference
                    unsafe {
                        UnsafeRef::into_box(cursor.remove().unwrap());
                    }
                    continue;
                }

                // Check the expiry for this entry, and if expired, skip it and continue
                //
                // If an entry is skipped here, it will be returned as part of the result list
                let expiration = entry.expiration.tick();
                if expiration.is_none() {
                    // This entry has expired, add it to the result list
                    self.timers.remove(&entry.timer.id());
                    // If this event is a recurring event, we need to schedule the next occurrance
                    if let Timer::Recurring { id, timeout, event } = &entry.timer {
                        let reschedule_entry = UnsafeRef::from_box(Box::new(TimerEntry::new(
                            Expiration::Overflow,
                            Timer::Recurring {
                                id: *id,
                                timeout: *timeout,
                                event: event.clone(),
                            },
                        )));
                        reschedule.push_back(reschedule_entry);
                    }
                    cursor.move_next();
                    continue;
                }

                // If we've reached this point, it's because the entry has not yet expired or been
                // cancelled, so we need to move it to another wheel based on its updated expiration
                // state.
                let mut entry = unsafe { UnsafeRef::into_box(cursor.remove().unwrap()) };
                entry.expiration = unsafe { expiration.unwrap_unchecked() };
                self.insert_to_wheel(entry);
            }

            // Splice all of the remaining entries in the list to the result list
            let mut cursor = result.back_mut();
            cursor.splice_after(entries);

            // If the wheel we just ticked has not yet completed a cycle, then we're done for now
            if !cycle_completed {
                return result;
            }

            // Check if this was the last wheel in the hierarchy; if so, then we need to proceed to the overflow entries
            if wheel == Self::LONG {
                break;
            }

            wheel += 1;
        }

        // If we reach here, then we've completed a full cycle and need to process the overflow entries
        let mut overflow = self.overflow.take();
        let current_time = self.current_time_in_cycle();
        while let Some(entry) = overflow.pop_front() {
            let mut entry = unsafe { UnsafeRef::into_box(entry) };

            if self.is_cancelled(&entry) {
                // The entry was removed from the timers map, so this is the last outstanding reference
                continue;
            }

            // If the timer is still too far in the future, send it back to the overflow list
            let timeout = entry.timer.timeout();
            if timeout >= Self::MAX_SCHEDULE_DURATION {
                self.insert_overflow(entry);
                continue;
            }

            // Otherwise, calculate
            match current_time.to_expiration(timeout) {
                Ok(expiration) => {
                    // We have an absolute expiration for this entry now, so place it in the wheel
                    entry.expiration = expiration;
                    self.insert_to_wheel(entry);
                }
                Err(_) => {
                    // This entry has expired, add it to the result list
                    self.timers.remove(&entry.timer.id());
                    // If this is a recurring timer, reschedule the next occurrance
                    //
                    // We can bypass the reschedule list here because we aren't holding a reference to the wheel
                    if let Timer::Recurring { id, timeout, event } = &entry.timer {
                        self.insert(Timer::Recurring {
                            id: *id,
                            timeout: *timeout,
                            event: event.clone(),
                        })
                        .unwrap();
                    }
                    result.push_back(UnsafeRef::from_box(entry));
                }
            }
        }

        // Handle rescheduling the recurring timers
        self.reschedule(reschedule);

        result
    }

    fn reschedule(&mut self, mut entries: TimerList) {
        let current_time = self.current_time_in_cycle();
        while let Some(entry) = entries.pop_front() {
            let mut entry = unsafe { UnsafeRef::into_box(entry) };

            // Handle a recurring timer with an overflow timeout
            let timeout = entry.timer.timeout();
            if timeout >= Self::MAX_SCHEDULE_DURATION {
                entry.expiration = Expiration::Overflow;
                self.insert_overflow(entry);
                continue;
            }

            // Otherwise, calculate the next expiration and insert in the wheel
            entry.expiration = current_time.to_expiration(timeout).unwrap();

            self.insert_to_wheel(entry);
        }
    }

    /// Attempts to skip `n` milliseconds into the future
    ///
    /// If successful, the wheel time is advanced, and `Ok` is returned.
    ///
    /// If there are timers that would have been fired in the skipped period, the wheel
    /// time is _not_ advanced, and `Err` is returned, containing the number of milliseconds
    /// that are actually skippable.
    pub fn try_skip(&mut self, n: u32) -> Result<(), u32> {
        if n == 0 {
            return Ok(());
        }

        match self.skippable() {
            None => Err(0),
            Some(skippable) if n <= skippable => {
                unsafe {
                    self.force_advance(n);
                }
                Ok(())
            }
            Some(skippable) => Err(skippable),
        }
    }

    /// Determines how much wheel time can be skipped until the next non-empty tick
    ///
    /// Returns `None` if no time can be skipped, otherwise it is a value which can be passed to `advance`
    pub fn skippable(&self) -> Option<u32> {
        let mut time = 0;
        let mut wheel = Self::SOON;
        let mut unit = Self::MS_PER_TICK_SOON;
        let mut empty = self.overflow.is_empty();

        loop {
            let skip = self.wheels[wheel].skippable();
            match skip {
                Skip::None => {
                    empty = false;
                    break;
                }
                Skip::Some(skippable) => {
                    empty = false;
                    time += skippable * unit;
                    break;
                }
                Skip::All | Skip::Empty => {
                    empty = empty && skip == Skip::Empty;
                    let current = self.wheels[wheel].current() as u32;
                    let skippable = (256 - current) - 1;
                    time += skippable * unit;
                    match wheel {
                        Self::SOON => {
                            wheel = Self::SHORT;
                            unit = Self::MS_PER_TICK_SHORT;
                        }
                        Self::SHORT => {
                            wheel = Self::MEDIUM;
                            unit = Self::MS_PER_TICK_MEDIUM;
                        }
                        Self::MEDIUM => {
                            wheel = Self::LONG;
                            unit = Self::MS_PER_TICK_LONG;
                        }
                        Self::LONG => break,
                        _ => unreachable!(),
                    }
                }
            }
        }

        if empty {
            Some((self.remaining_time_in_cycle() - 1) as u32)
        } else {
            match time {
                0 => None,
                n => Some(n),
            }
        }
    }

    /// Advances the wheel time until the next tick at which a timer may expire.
    ///
    /// If the wheel is empty, this function will leave the wheel time reset to the beginning of a new cycle.
    #[cfg(test)]
    #[allow(unused)]
    pub fn skip(&mut self) {
        if !self.wheels[Self::SOON].skip() {
            return;
        }

        if !self.wheels[Self::SHORT].skip() {
            return;
        }

        if !self.wheels[Self::MEDIUM].skip() {
            return;
        }

        if !self.wheels[Self::LONG].skip() {
            return;
        }

        if self.overflow.is_empty() {
            unsafe {
                self.wheels[Self::SOON].advance(0);
                self.wheels[Self::SHORT].advance(0);
                self.wheels[Self::MEDIUM].advance(0);
                self.wheels[Self::LONG].advance(0);
            }
        }
    }

    /// Advances the wheel time by `n` milliseconds
    ///
    /// # SAFETY
    ///
    /// This function bypasses all of the timers which would have expired during the
    /// skipped time period. As a result, none of their corresponding events will be fired.
    ///
    /// This is a low-level operation intended for use by functions which can guarantee skips
    /// don't violate any invariants.
    pub unsafe fn force_advance(&mut self, n: u32) {
        let new_time = self.current_time_in_cycle().wrapping_add(n);
        self.wheels[Self::SOON].advance(new_time.soon());
        self.wheels[Self::SHORT].advance(new_time.short());
        self.wheels[Self::MEDIUM].advance(new_time.medium());
        self.wheels[Self::LONG].advance(new_time.long());
    }

    fn remaining_time_in_cycle(&self) -> u64 {
        Self::CYCLE_LENGTH - (self.current_time_in_cycle().as_u32() as u64)
    }

    fn current_time_in_cycle(&self) -> Time {
        Time::new(
            self.wheels[Self::SOON].current(),
            self.wheels[Self::SHORT].current(),
            self.wheels[Self::MEDIUM].current(),
            self.wheels[Self::LONG].current(),
        )
    }

    fn expiration_to_wheel_and_slot(&self, expiration: Expiration) -> Option<(usize, usize)> {
        match expiration {
            Expiration::Soon(time) => Some((Self::SOON, time.soon() as usize)),
            Expiration::Short(time) => Some((Self::SHORT, time.short() as usize)),
            Expiration::Medium(time) => Some((Self::MEDIUM, time.medium() as usize)),
            Expiration::Long(time) => Some((Self::LONG, time.long() as usize)),
            Expiration::Overflow => None,
        }
    }

    fn is_cancelled(&self, entry: &TimerEntry) -> bool {
        // If the timer is no longer tracked, it was cancelled
        !self.timers.contains_key(&entry.timer.id())
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum Skip {
    // The wheel is empty, so skipping is a no-op
    Empty,
    // Nothing can be skipped
    None,
    // The given number of slots can be skipped this cycle before hitting a non-skippable slot
    Some(u32),
    // The rest of this cycle can be skipped
    All,
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;
    use core::sync::atomic::{AtomicBool, Ordering};

    use firefly_system::time::Timeout;

    use crate::scheduler::SchedulerId;
    use crate::services::timers::{Timer, TimerEvent};
    use crate::term::ReferenceId;

    use super::*;

    #[test]
    fn hierarchical_wheel_start_timer_test() {
        let mut wheel = HierarchicalTimerWheel::new();

        static SUCCESS: AtomicBool = AtomicBool::new(false);

        let timer_ref = unsafe { ReferenceId::new(SchedulerId::from_raw(0), 1) };
        let timer = Timer::Once {
            id: timer_ref,
            timeout: Timeout::from_millis(5),
            event: TimerEvent::Callback(Box::new(|| {
                SUCCESS
                    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
                    .unwrap();
            })),
        };

        wheel.insert(timer).unwrap();

        // Skip right up to just before our timer should expire, no timers should have been triggered
        assert_eq!(Ok(()), wheel.try_skip(4));

        let entries = wheel.tick();
        assert_eq!(entries.iter().count(), 0);

        // Try to skip just past when our timer should expire, we shouldn't be able to
        assert_eq!(Err(0), wheel.try_skip(1));

        let mut entries = wheel.tick();
        assert_eq!(entries.iter().count(), 1);
        while let Some(entry) = entries.pop_front() {
            match unsafe { UnsafeRef::into_box(entry) }.into_timer() {
                Timer::Once { event, .. } => match event {
                    TimerEvent::Callback(callback) => callback(),
                    _ => panic!("unexpected timer event"),
                },
                _ => panic!("unexpected timer type"),
            }
        }

        assert!(SUCCESS.load(Ordering::SeqCst));
    }

    #[test]
    fn hierarchical_wheel_cancel_timer_test() {
        let mut wheel = HierarchicalTimerWheel::new();

        static SUCCESS: AtomicBool = AtomicBool::new(true);

        let timer_ref = unsafe { ReferenceId::new(SchedulerId::from_raw(0), 1) };
        let timer = Timer::Once {
            id: timer_ref,
            timeout: Timeout::from_millis(5),
            event: TimerEvent::Callback(Box::new(|| {
                SUCCESS
                    .compare_exchange(true, false, Ordering::SeqCst, Ordering::SeqCst)
                    .unwrap();
            })),
        };

        wheel.insert(timer).unwrap();

        assert_eq!(Ok(()), wheel.try_skip(4));

        let entries = wheel.tick();
        assert_eq!(entries.iter().count(), 0);

        assert_eq!(Ok(()), wheel.cancel(timer_ref));

        assert_eq!(Ok(()), wheel.try_skip(2));

        let entries = wheel.tick();
        assert_eq!(entries.iter().count(), 0);

        assert!(SUCCESS.load(Ordering::SeqCst));
    }
}

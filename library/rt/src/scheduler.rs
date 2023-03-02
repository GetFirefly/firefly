use alloc::sync::Arc;
use core::mem::MaybeUninit;

use firefly_number::Int;
use firefly_system::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::function::ModuleFunctionArity;
use crate::gc::Gc;
use crate::process::{Process, ProcessLock, SpawnOpts};
use crate::services::timers::{Timer, TimerError};
use crate::term::{OpaqueTerm, Reference, ReferenceId};

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SchedulerId(u16);
impl SchedulerId {
    pub const INVALID: Self = Self(u16::MAX);

    /// Reify a `SchedulerId` from it's raw representation
    ///
    /// # SAFETY
    ///
    /// This function must never be called on a value which is not known
    /// to be a valid active scheduler id. If used improperly, it could result
    /// in multiple schedulers with the same id; incorrectly associating a
    /// value with the wrong scheduler; or producing a reference for which no
    /// scheduler actually exists.
    #[inline]
    pub(crate) unsafe fn from_raw(id: u16) -> Self {
        Self(id)
    }

    #[inline(always)]
    pub const fn as_u16(&self) -> u16 {
        self.0
    }

    /// Returns true if this scheduler id is invalid, meaning it was not assigned by
    /// the runtime scheduler set, and is a default value
    #[inline(always)]
    pub fn is_invalid(&self) -> bool {
        self == &Self::INVALID
    }
}
impl Default for SchedulerId {
    #[inline(always)]
    fn default() -> Self {
        Self::INVALID
    }
}
impl Into<u16> for SchedulerId {
    #[inline(always)]
    fn into(self) -> u16 {
        self.0
    }
}
impl firefly_system::sync::Atom for SchedulerId {
    type Repr = u16;

    #[inline]
    fn pack(self) -> Self::Repr {
        self.as_u16()
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        unsafe { Self::from_raw(raw) }
    }
}

pub trait Scheduler: Send + Sync {
    /// The unique id of this scheduler
    fn id(&self) -> SchedulerId;

    /// The thread id of this scheduler
    ///
    /// This must match the thread in which the scheduler is executing
    #[cfg(feature = "std")]
    fn thread_id(&self) -> std::thread::ThreadId;

    /// Generates a new [`ReferenceId`], unique for this scheduler
    ///
    /// The resulting identifier can be used with `Reference` functions which expect one
    fn next_reference_id(&self) -> ReferenceId;

    /// Generates a new unique integer value for this scheduler
    ///
    /// If `positive` is true, the value will be interpreted as an unsigned integer.
    /// If it is false, then it will be interpreted as a signed integer.
    fn next_unique_integer(&self, positive: bool) -> Int;

    /// Generates the next monotonically-increasing unique integer value for this scheduler
    ///
    /// If `positive` is true, the value will be interpreted as an unsigned integer.
    /// If it is false, then it will be interpreted as a signed integer.
    fn next_monotonic_integer(&self, positive: bool) -> Int;

    /// Request this scheduler to start `timer` via its timer service
    fn start_timer(&self, timer: Timer) -> Result<(), TimerError>;

    /// Request to cancel a timer previously started via `start_timer`
    ///
    /// Returns `Ok` if successful, `Err` if no such timer exists
    fn cancel_timer(&self, timer_ref: ReferenceId) -> Result<(), ()>;

    /// Spawn a new process with the given module/function/arguments
    ///
    /// The spawned process will exit with an error if the given MFA is invalid, or the arguments
    /// are incorrect for the callee.
    ///
    /// The result contains a reference to the spawned process, and if being monitored, the monitor
    /// reference.
    ///
    /// NOTE: Callers must ensure there is sufficient space on the process heap for the
    /// monitor ref if applicable. If not, this function will panic.
    fn spawn(
        &self,
        parent: &mut ProcessLock,
        mfa: ModuleFunctionArity,
        args: &[OpaqueTerm],
        opts: SpawnOpts,
    ) -> (Arc<Process>, Option<Gc<Reference>>);

    /// Reschedules `process` using this scheduler's run queue
    fn reschedule(&self, process: Arc<Process>);
}

/// Returns a strong reference to the scheduler corresponding to `id`
///
/// This function will panic if the id is invalid, or the scheduler is not available
pub fn get(id: SchedulerId) -> Arc<dyn Scheduler> {
    with_schedulers_readonly(|schedulers| schedulers.fetch(id))
}

/// Creates a new scheduler using the provided constructor function.
///
/// The function provided can expect a scheduler id to be provided which is safe for use
/// by the scheduler, and should return an `Arc<dyn Scheduler>` representing the initialized
/// scheduler.
///
/// The scheduler will be considered "online" when this function returns, but it is up to the
/// caller to actually start running the scheduler loop.
pub fn create<S, F, E>(ctor: F) -> Result<Arc<S>, E>
where
    S: Scheduler + 'static,
    F: FnOnce(SchedulerId) -> Result<Arc<S>, E>,
{
    // Allocate a scheduler id
    let id = with_schedulers(|mut schedulers| schedulers.next_id());
    // Run the constructor while not holding the lock
    match ctor(id) {
        // If successful, register the scheduler
        Ok(scheduler) => {
            let dyn_scheduler = scheduler.clone() as Arc<dyn Scheduler>;
            with_schedulers(move |mut schedulers| schedulers.insert(dyn_scheduler));
            Ok(scheduler)
        }
        // If successful, release the scheduler id we were assigned
        Err(error) => {
            with_schedulers(|mut schedulers| unsafe { schedulers.release(id) });
            Err(error)
        }
    }
}

/// Removes a scheduler from the offline set, and releases its id back to the pool
///
/// This must only be called by the scheduler itself, but because the scheduler set holds
/// a reference, it isn't possible for us to guarantee that the caller has exclusive control
/// of the scheduler.
pub fn remove(scheduler: &dyn Scheduler) {
    with_schedulers(|mut schedulers| unsafe { schedulers.remove(scheduler.id()) })
}

/// Returns the number of schedulers online
pub fn online() -> u32 {
    with_schedulers_readonly(|schedulers| schedulers.online())
}

/// Returns the number of schedulers online
pub fn dirty_cpu() -> u32 {
    with_schedulers_readonly(|schedulers| schedulers.dirty_cpu())
}

/// Returns the number of schedulers online
pub fn dirty_io() -> u32 {
    with_schedulers_readonly(|schedulers| schedulers.dirty_io())
}

#[inline]
fn with_schedulers_readonly<F, T>(callback: F) -> T
where
    F: FnOnce(RwLockReadGuard<'static, SchedulerSet>) -> T,
{
    let schedulers = SCHEDULERS.get_or_init(|| RwLock::new(SchedulerSet::new()));
    callback(schedulers.read())
}

#[inline]
fn with_schedulers<F, T>(callback: F) -> T
where
    F: FnOnce(RwLockWriteGuard<'static, SchedulerSet>) -> T,
{
    let schedulers = SCHEDULERS.get_or_init(|| RwLock::new(SchedulerSet::new()));
    callback(schedulers.write())
}

static SCHEDULERS: OnceLock<RwLock<SchedulerSet>> = OnceLock::new();

struct SchedulerSet {
    /// The set of schedulers, up to 64 can be run at the same time in this configuration
    ///
    /// Each element in this array is initially `None`, but can be safely assumed `Some` if
    /// the bit corresponding to the element index in the `online` bitset is set.
    schedulers: [Option<Arc<dyn Scheduler>>; 64],
    /// A bitset defining which schedulers are online/initialized
    online: u64,
    /// A bitset defining which scheduler slots are available for assignment
    assigned: u64,
}
impl SchedulerSet {
    const MAX_SCHEDULERS: usize = 64;

    pub fn new() -> Self {
        let mut slots = MaybeUninit::<Option<Arc<dyn Scheduler>>>::uninit_array::<64>();
        for slot in &mut slots {
            slot.write(None);
        }

        Self {
            schedulers: unsafe { MaybeUninit::array_assume_init(slots) },
            online: 0,
            assigned: 0,
        }
    }

    /// Acquire the next available scheduler slot/id
    ///
    /// This marks the given id as used.
    pub fn next_id(&mut self) -> SchedulerId {
        let next_free = self.assigned.trailing_zeros();
        assert!(
            next_free > 0,
            "system limit: reached the maximum number of schedulers online"
        );

        // Mark the slot as assigned
        self.assigned |= 1 << (next_free - 1);

        SchedulerId((Self::MAX_SCHEDULERS - next_free as usize) as u16)
    }

    /// Inserts `scheduler` in the set, marking it as online
    ///
    /// The given scheduler must have an id assigned by this scheduler set, otherwise this function
    /// will panic. See [`SchedulerId::next`].
    pub fn insert(&mut self, scheduler: Arc<dyn Scheduler>) {
        let id = scheduler.id().as_u16() as usize;
        assert!(
            id < Self::MAX_SCHEDULERS,
            "invalid scheduler id, make sure you request a scheduler id from scheduler set"
        );

        // Make sure the scheduler id is claimed and not online
        let id_bit = 1u64 << (id as u64);
        assert_eq!(
            self.assigned & id_bit,
            0,
            "invalid scheduler id, was not assigned by the scheduler set"
        );
        assert_eq!(
            self.online & id_bit,
            0,
            "invalid scheduler id, was not assigned by the scheduler set"
        );

        // Mark the scheduler online
        self.online |= id_bit;

        let slot = unsafe { self.schedulers.get_unchecked_mut(id) };
        assert!(slot.replace(scheduler).is_none());
    }

    /// Releases a previously assigned scheduler identifier, `id`.
    ///
    /// This is only safe to call when an id has been assigned, but is not actively used by an
    /// online scheduler
    pub unsafe fn release(&mut self, id: SchedulerId) {
        let id = id.as_u16() as usize;
        assert!(
            id < Self::MAX_SCHEDULERS,
            "invalid scheduler id, make sure you request a scheduler id from scheduler set"
        );
        let bit = 1 << (id as u64);
        assert_eq!(
            self.assigned & bit,
            bit,
            "cannot release a scheduler id that was not assigned"
        );
        assert_eq!(
            self.online & bit,
            0,
            "cannot release a scheduler id that is still online"
        );
        self.assigned &= !bit;
    }

    /// Like `release`, but for cases in which the scheduler was online and is now terminating.
    ///
    /// This marks the scheduler as offline, frees the scheduler id, and drops the reference we
    /// hold.
    pub unsafe fn remove(&mut self, id: SchedulerId) {
        let id = id.as_u16() as usize;
        assert!(
            id < Self::MAX_SCHEDULERS,
            "invalid scheduler id, make sure you request a scheduler id from scheduler set"
        );
        let bit = 1 << (id as u64);
        assert_eq!(
            self.assigned & bit,
            bit,
            "cannot remove a scheduler that was not assigned"
        );
        assert_eq!(
            self.online & bit,
            0,
            "cannot remove a scheduler that is not online"
        );
        self.online &= !bit;
        self.assigned &= !bit;
        let slot = unsafe { self.schedulers.get_unchecked_mut(id) };
        slot.take();
    }

    /// Retrieves the scheduler with the given identifier
    ///
    /// This function will panic if the scheduler id is invalid, or if the requested scheduler is
    /// not online
    pub fn fetch(&self, id: SchedulerId) -> Arc<dyn Scheduler> {
        let id = id.as_u16() as usize;
        assert!(
            id < Self::MAX_SCHEDULERS,
            "invalid scheduler id, make sure you request a scheduler id from scheduler set"
        );

        // Make sure the scheduler is online
        let id_bit = 1u64 << (id as u64);
        assert_eq!(
            self.online & id_bit,
            id_bit,
            "invalid scheduler id, scheduler is not online"
        );

        let slot = unsafe { self.schedulers.get_unchecked(id) };
        slot.clone().unwrap()
    }

    /// Returns true if the scheduler with the given `id` is online
    #[allow(unused)]
    pub fn is_online(&self, id: SchedulerId) -> bool {
        let bit = 1 << (id.as_u16() as u64);
        self.online & bit == bit
    }

    /// Returns the number of schedulers online
    pub fn online(&self) -> u32 {
        self.online.count_ones()
    }

    /// Returns the number of dirty CPU schedulers online
    pub fn dirty_cpu(&self) -> u32 {
        0
    }

    /// Returns the number of dirty I/O schedulers online
    pub fn dirty_io(&self) -> u32 {
        0
    }
}

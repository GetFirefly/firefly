mod wheel;

pub use self::wheel::TimerList;
use self::wheel::{HierarchicalTimerWheel, TimerEntry};

use alloc::alloc::AllocError;
use alloc::boxed::Box;
use alloc::sync::{Arc, Weak};
use core::fmt;

use firefly_alloc::fragment::HeapFragment;
use firefly_system::time::{MonotonicTime, Timeout};

use intrusive_collections::UnsafeRef;

use log::trace;

use crate::gc::Gc;
use crate::process::{Process, ProcessLock, ProcessTimer};
use crate::services::registry::{Registrant, WeakAddress};
use crate::term::{atoms, LayoutBuilder, Reference, ReferenceId, Term, TermFragment, Tuple};

/// Represents errors which can occur when interacting with the timer wheel
#[derive(Debug)]
pub enum TimerError {
    /// Infinite timeouts should never reach the timer wheel
    InvalidTimeout(Timer),
    /// The timer given has already expired
    Expired(Timer),
    /// Failed to allocate space for the timer message
    Alloc,
}

/// Timers are essentially programmable events with a time component.
///
/// They can be configured to fire on either a one-time or recurring basis,
/// support cancellation, and are uniquely identifiable using `Reference` identifiers.
///
/// The time component of a timer is a [`Timeout`] value, from which the absolute expiration
/// time of the timer is derived. In the case of recurring timers, this expresses the interval
/// of the timer.
///
/// Timers currently only support a specific set of of events to fire when a timer expires,
/// see [`TimerEvent`] for details.
pub enum Timer {
    /// A one-shot timer, commonly used for things like timing out a blocking operation.
    Once {
        id: ReferenceId,
        timeout: Timeout,
        event: TimerEvent,
    },
    /// A periodic timer, intended to support recurring background tasks
    Recurring {
        id: ReferenceId,
        timeout: Timeout,
        event: Arc<dyn Fn(ReferenceId) + Send + 'static>,
    },
}
impl fmt::Debug for Timer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Once { id, timeout, event } => f
                .debug_struct("Once")
                .field("id", id)
                .field("timeout", timeout)
                .field("event", event)
                .finish(),
            Self::Recurring { id, timeout, .. } => f
                .debug_struct("Recurring")
                .field("id", id)
                .field("timeout", timeout)
                .finish(),
        }
    }
}
impl Timer {
    pub fn id(&self) -> ReferenceId {
        match self {
            Self::Once { id, .. } | Self::Recurring { id, .. } => *id,
        }
    }

    pub fn timeout(&self) -> Timeout {
        match self {
            Self::Once { timeout, .. } | Self::Recurring { timeout, .. } => *timeout,
        }
    }

    fn timeout_mut(&mut self) -> &mut Timeout {
        match self {
            Self::Once {
                ref mut timeout, ..
            }
            | Self::Recurring {
                ref mut timeout, ..
            } => timeout,
        }
    }

    pub fn is_recurring(&self) -> bool {
        match self {
            Self::Recurring { .. } => true,
            _ => false,
        }
    }
}

/// The event which occurs when a one-shot timer fires
pub enum TimerEvent {
    /// Sends `message` to `recipient`
    Message {
        sender: WeakAddress,
        recipient: WeakAddress,
        message: TermFragment,
    },
    /// Notifies a process that the operation on which it was waiting has timed out.
    ///
    /// When this event is fired, the process may have already been rescheduled or exited
    /// as a result of other events in the system, so this only has an effect if the process
    /// is still in a waiting state.
    ///
    /// NOTE: We must make sure elsewhere that timers associated with blocking operations
    /// are cancelled when resuming from the waiting state, or we risk timing out an unrelated
    /// operation due to race conditions.
    Timeout(Weak<Process>),
    /// Calls the given callback
    Callback(Box<dyn FnOnce() + Send + 'static>),
}
impl fmt::Debug for TimerEvent {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Message {
                sender,
                recipient,
                message,
            } => f
                .debug_struct("Message")
                .field("sender", sender)
                .field("recipient", recipient)
                .field("message", &message.term)
                .finish(),
            Self::Timeout(_) => write!(f, "Timeout(Weak)"),
            Self::Callback(_) => write!(f, "Callback"),
        }
    }
}

/// This trait provides the minimum interface required for implementations of the runtime timer service.
///
/// Timer services might be designed for use a couple different ways:
///
/// * A single global instance, where the timer service runs in a dedicated thread and accept requests over some kind of MPSC queue.
/// This design means mutable references are not possible in its interface.
/// * Per-scheduler instances, where the timer service is owned by a specific scheduler, and timer operations are single-threaded.
/// This design means mutable references can be used, and are actually preferable for performance reasons.
///
/// This trait is oriented towards the latter design, i.e. it uses mutable references in its interface.
///
/// In the future, we may decide to swing the other way, or introduce another trait for global timer services.
pub trait TimerService {
    /// Starts the timer service
    ///
    /// This may mean different things to different implementations, but generally this function
    /// should set up whatever is necessary to drive its internal timer wheel, and return `Ok`.
    /// Subsequent calls should return `Err` to indicate that the service is already started.
    fn start(&mut self) -> Result<(), ()>;

    /// Stops the timer service
    ///
    /// When this occurs and there are any outstanding timers, they are implicitly canceled.
    fn stop(&mut self);

    /// Starts `timer` via this service
    ///
    /// Returns `Err` if the timer has an invalid timeout (i.e. infinite) or is effectively expired.
    ///
    /// Otherwise, `Ok` indicates that the timer was successfully started
    ///
    /// NOTE: If the timer service is not yet started, this must return `Err`
    fn start_timer(&mut self, timer: Timer) -> Result<(), TimerError>;

    /// Starts a new `timer` as a process timer via this service
    ///
    /// Returns `Err` if the timer has an invalid timeout (i.e. infinite) or is effectively expired.
    ///
    /// Otherwise, `Ok` indicates that the timer was successfully started
    ///
    /// NOTE: If the timer service is not yet started, this must return `Err`
    fn start_process_timer(
        &mut self,
        process: &mut ProcessLock,
        timer: Timer,
    ) -> Result<(), TimerError>;

    /// Cancels a timer identified by `timer_ref`
    ///
    /// Returns `Err` if the timer could not be found.
    ///
    /// Returns `Ok` if the timer was successfully canceled.
    fn cancel_timer(&mut self, timer_ref: ReferenceId) -> Result<(), ()>;

    /// Creates and starts a new one-shot [`Timer`] which sends `message` to `recipient` after `timeout`.
    ///
    /// The provided `timer_ref` will be used as the reference for the created timer.
    fn send_after(
        &mut self,
        timer_ref: ReferenceId,
        process: Arc<Process>,
        message: TermFragment,
        recipient: WeakAddress,
        timeout: Timeout,
    ) -> Result<(), TimerError> {
        let timer = Timer::Once {
            id: timer_ref,
            timeout,
            event: TimerEvent::Message {
                sender: process.pid().into(),
                recipient,
                message,
            },
        };
        self.start_timer(timer)
    }

    /// Creates a new one-shot [`Timer`] which sends `message` to `recipient` after `timeout`.
    ///
    /// The provided `timer_ref` will be used as the reference for the created timer.
    ///
    /// This differs from `send_after`, in that the given message is formatted as `{timeout, Reference, Message}`
    /// when sent to the process, where `Reference` is the timer reference, and `Message` is the original `message`
    /// term.
    fn send_timeout_after(
        &mut self,
        timer_ref: ReferenceId,
        process: Arc<Process>,
        message: Term,
        recipient: WeakAddress,
        timeout: Timeout,
    ) -> Result<(), TimerError> {
        let fragment = timeout_triple(timer_ref, message).map_err(|_| TimerError::Alloc)?;
        self.send_after(timer_ref, process, fragment, recipient, timeout)
    }

    /// Creates a new one-shot [`Timer`] which notifies `process` that the operation it was waiting on has timed out.
    ///
    /// The provided `timer_ref` will be used as the reference for the created timer.
    ///
    /// This differs from `send_after` and `send_timeout_after` in that no message is being sent. Instead, the process
    /// is being signaled that the last operation it was waiting on should be cancelled. This only has an effect if the
    /// process is still alive, and still in the waiting state.
    ///
    /// NOTE: The resulting timer must be inserted in a timer wheel to be activated
    fn timeout_after(
        &mut self,
        timer_ref: ReferenceId,
        process: &mut ProcessLock,
        timeout: Timeout,
    ) -> Result<(), TimerError> {
        let timer = Timer::Once {
            id: timer_ref,
            timeout,
            event: TimerEvent::Timeout(process.weak()),
        };
        self.start_process_timer(process, timer)
    }
}

/// This is a simple implementation of the [`TimerService`] trait for per-scheduler use.
///
/// This implementation is designed to be driven by a scheduler as part of its core loop.
/// On each iteration of the loop, `tick` should be called to drive the timer wheel forward
/// an amount corresponding to the elapsed time in milliseconds since the last call to `tick`.
///
/// This implementation also supports arbitrarily skipping forward in time to the next event
/// in the wheel, which is useful for tests which need to interact with a timer service without
/// needing to run in real-time.
///
/// This service is implicitly started on creation, so calls to `start` always return `Err`, and
/// `stop` is a no-op.
pub struct PerSchedulerTimerService {
    wheel: HierarchicalTimerWheel,
    /// The monotonic timestamp of the last call to `tick`.
    ///
    /// Defaults to the start time of the service.
    last_tick: MonotonicTime,
}
impl PerSchedulerTimerService {
    pub fn new() -> Self {
        Self {
            wheel: HierarchicalTimerWheel::new(),
            last_tick: MonotonicTime::now(),
        }
    }
}
impl TimerService for PerSchedulerTimerService {
    fn start(&mut self) -> Result<(), ()> {
        Err(())
    }

    fn stop(&mut self) {}

    #[inline]
    fn start_timer(&mut self, timer: Timer) -> Result<(), TimerError> {
        self.wheel.insert(timer).map(|_| ())
    }

    #[inline]
    fn start_process_timer(
        &mut self,
        process: &mut ProcessLock,
        timer: Timer,
    ) -> Result<(), TimerError> {
        let timer_ref = timer.id();
        let entry_ptr = self.wheel.insert(timer)?;
        process
            .set_timer(
                ProcessTimer::Active(entry_ptr.as_ptr() as *const ()),
                timer_ref,
            )
            .unwrap();
        Ok(())
    }

    #[inline]
    fn cancel_timer(&mut self, timer_ref: ReferenceId) -> Result<(), ()> {
        self.wheel.cancel(timer_ref)
    }
}
impl PerSchedulerTimerService {
    /// Returns true if there are no timers registered
    pub fn is_empty(&self) -> bool {
        self.wheel.is_empty()
    }

    /// Determines how much wheel time can be skipped until the next non-empty tick
    ///
    /// Returns `None` if no time can be skipped, otherwise it is the number of milliseconds that can be skipped
    pub fn skippable(&self) -> Option<u32> {
        self.wheel.skippable()
    }

    /// Computes the elapsed time relative to the monotonic system clock, and ticks the wheel forward to
    /// that point, firing any timers that fall within the elapsed period.
    ///
    /// Returns `true` if events occurred during this tick
    pub fn tick(&mut self) -> bool {
        let now = MonotonicTime::now();
        trace!(target: "timers", "tick started at {}", now);
        trace!(target: "timers", "last tick started at {}", self.last_tick);
        let elapsed_time = now - self.last_tick;
        let mut elapsed_ms: u32 = elapsed_time.as_millis().try_into().unwrap();
        trace!(target: "timers", "{}ms elapsed since last tick", elapsed_ms);
        self.last_tick = now;

        // The following code is retried multiple times until the elapsed time is passed
        let mut events_fired = false;
        loop {
            // When we've advanced the wheel by the time elapsed, we're done
            if elapsed_ms == 0 {
                break;
            }

            // Attempt to fast forward the clock
            match self.wheel.try_skip(elapsed_ms) {
                // No timers occurred during the elapsed period, and the clock
                // has been advanced to the new time, we're done.
                Ok(_) => {
                    break;
                }
                // We can't skip the entire elapsed period, but we can skip some of it
                //
                // This will leave the wheel in position to produce entries on the next tick
                Err(skippable) if skippable > 0 => {
                    elapsed_ms -= skippable;
                    unsafe {
                        self.wheel.force_advance(skippable);
                    }
                }
                // We can't skip anything
                Err(_) => (),
            }

            trace!(target: "timers", "one or more timers timed out during the elapsed period");
            // If we reach here, we have timer entries to fire
            let mut entries = self.wheel.tick();
            events_fired = events_fired || !entries.is_empty();
            while let Some(entry) = entries.pop_front() {
                let ptr = entry.as_ref() as *const TimerEntry as *const ();
                self.fire(ptr, unsafe { UnsafeRef::into_box(entry) }.into_timer());
            }

            // When we reach here, elapsed_ms is always positive
            elapsed_ms -= 1;
        }

        events_fired
    }

    fn fire(&self, id: *const (), timer: Timer) {
        match timer {
            Timer::Once { event, .. } => match event {
                TimerEvent::Timeout(weak) => {
                    if let Some(process) = weak.upgrade() {
                        trace!(target: "timers", "process timeout expired for {}", process.pid());
                        // This should always be uncontested, since the process is suspended
                        // awaiting this timeout.
                        if let Err(_) = process.set_timeout(ProcessTimer::Active(id)) {
                            // Timer was cancelled or cancelled/replaced, do nothing
                            trace!(target: "timers", "process timeout was already cancelled/replaced");
                            return;
                        }
                    }
                }
                TimerEvent::Callback(callback) => {
                    trace!(target: "timers", "callback timer expired");
                    callback()
                }
                TimerEvent::Message {
                    sender,
                    recipient,
                    message,
                } => match recipient.try_resolve() {
                    None => {
                        trace!(target: "timers", "send_after timer expired, but process is dead");
                    }
                    Some(Registrant::Process(process)) => {
                        trace!(target: "timers", "send_after timer expired for {}", process.pid());
                        process.send_fragment(sender, message).ok();
                    }
                    _ => unimplemented!(),
                },
            },
            Timer::Recurring { id, event, .. } => {
                trace!(target: "timers", "recurring timer {} expired", id);
                event(id);
            }
        }
    }
}

/// Allocates a new [`TermFragment`] containing `message` wrapped in a timeout triple.
///
/// The timeout triple format is `{timeout, Reference, Message}`, and is used by the `send_timeout_after`
/// function which is specifically designed for sending timeout messages in this format.
pub fn timeout_triple(timer_ref: ReferenceId, message: Term) -> Result<TermFragment, AllocError> {
    // The timeout triple requires the following layout:
    //
    // | message (if boxed, immediates live in the tuple)
    // | reference
    // | tuple
    // |   atom
    // |   box (reference)
    // |   immediate or box (message)
    // * size_of<Tuple>
    let mut builder = LayoutBuilder::new();
    builder.extend(&message).build_reference().build_tuple(3);
    let fragment = HeapFragment::new(builder.finish(), None)?;
    let heap = unsafe { fragment.as_ref() };
    let cloned = message.clone_to_heap(heap)?;
    let reference = Gc::new_in(Reference::new(timer_ref), heap)?;
    let tag = atoms::Timeout.into();
    let message = cloned.into();
    let tuple = Tuple::from_slice(&[tag, reference.into(), message], &heap)?;
    Ok(TermFragment {
        term: tuple.into(),
        fragment: Some(fragment),
    })
}

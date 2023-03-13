use alloc::boxed::Box;
use alloc::sync::Arc;
use core::any::Any;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::intrinsics::unlikely;
use core::mem::{self, MaybeUninit};
use core::num::NonZeroU64;
use core::ops::{Deref, DerefMut};
use core::ptr;
use core::sync::atomic::{AtomicUsize, Ordering};

use intrusive_collections::intrusive_adapter;
use intrusive_collections::{LinkedList, LinkedListLink};

use log::trace;

use firefly_system::mem::CachePadded;
use firefly_system::sync::{Atomic, Mutex, MutexGuard};

use crate::services::registry::WeakAddress;
use crate::term::{Atom, Pid, Reference, TermFragment};

use super::link::LinkEntry;
use super::monitor::MonitorEntry;
use super::{Priority, Process, ProcessLock};

pub type RpcCallback = fn(process: &mut ProcessLock, state: *mut ()) -> TermFragment;

/// A trait for signals which are produced by the runtime system
///
/// Signals may be system-generated, or sent by a process or port, and are received by a process.
/// It is possible for the sender and receiver to be the same process.
///
/// The receiving process is always the one handling a signal.
pub trait DynSignal: Any + Send {
    /// The address of the sender of this signal
    ///
    /// Some signals might be sent by the system itself, or might have no sender
    /// available, such as when a link is broken by a disconnection in the distribution
    /// subsystem, in which case there is no address.
    fn sender(&self) -> Option<WeakAddress>;
}

/// Represents the type of flush to perform on the signal queue
#[derive(Debug, PartialEq, Eq)]
pub enum FlushType {
    /// Flush the in-transit queue to the private queue
    InTransit,
    /// Flush signals from all local senders (processes and ports)
    Local,
    /// Flush signals from the process or port identified by the given address
    Id(WeakAddress),
}

/// The set of recognized signal types
pub enum Signal {
    /// A message was sent
    Message(Message),
    /// An exit signal was sent
    Exit(Exit),
    /// An exit signal due to a broken link
    ExitLink(Exit),
    /// A monitor was created by the sender for the receiver
    Monitor(Monitor),
    /// A monitor was removed by the sender for the receiver
    Demonitor(Demonitor),
    /// A monitor owned by the receiver, was triggered due to the sender exiting
    MonitorDown(MonitorDown),
    /// A link was created by the sender for the receiver
    Link(Link),
    /// A request to remove a previously established link was made by the sender to the receiver
    Unlink(Unlink),
    /// An unlink request was acknowledged by the sender to the receiver
    UnlinkAck(UnlinkAck),
    /// The sender is changing the group leader of the receiver to some other process
    GroupLeader(GroupLeader),
    /// The sender is requesting the receiver to reply with a message indicating it is alive or not
    IsAlive(IsAlive),
    /// The sender is requesting process information about the receiver
    ProcessInfo(ProcessInfo),
    /// Initiates a flush of the signal queue when received
    Flush(Flush),
    /// Executes a function in the context of the receiving process.
    ///
    /// Sends a reply of `{Ref, Result}` with the value returned from the function.
    ///
    /// Because the receiver may execute its function while exiting, senders of this message
    /// type must unconditionally enter a receive that matches on `Ref` in all clauses, or bad
    /// things will happen.
    Rpc(Rpc),
}
impl fmt::Debug for Signal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Message(_) => f.debug_struct("Message").finish(),
            Self::Exit(_) => f.debug_struct("Exit").finish(),
            Self::ExitLink(_) => f.debug_struct("ExitLink").finish(),
            Self::Monitor(_) => f.debug_struct("Monitor").finish(),
            Self::Demonitor(_) => f.debug_struct("Demonitor").finish(),
            Self::MonitorDown(_) => f.debug_struct("MonitorDown").finish(),
            Self::Link(_) => f.debug_struct("Link").finish(),
            Self::Unlink(_) => f.debug_struct("Unlink").finish(),
            Self::UnlinkAck(_) => f.debug_struct("UnlinkAck").finish(),
            Self::GroupLeader(_) => f.debug_struct("GroupLeader").finish(),
            Self::IsAlive(_) => f.debug_struct("IsAlive").finish(),
            Self::ProcessInfo(_) => f.debug_struct("ProcessInfo").finish(),
            Self::Flush(_) => f.debug_struct("Flush").finish(),
            Self::Rpc(_) => f.debug_struct("Rpc").finish(),
        }
    }
}
impl Signal {
    pub fn is_process_info(&self) -> bool {
        match self {
            Self::ProcessInfo(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn message(sender: WeakAddress, message: TermFragment) -> Box<SignalEntry> {
        SignalEntry::new(Self::Message(Message { sender, message }))
    }

    #[inline]
    pub fn monitor(monitor: Arc<MonitorEntry>) -> Box<SignalEntry> {
        SignalEntry::new(Self::Monitor(Monitor { monitor }))
    }

    #[inline]
    pub fn unlink(sender: WeakAddress, id: NonZeroU64) -> Box<SignalEntry> {
        SignalEntry::new(Self::Unlink(Unlink { sender, id }))
    }

    #[inline]
    pub fn is_alive(sender: Pid, reference: Reference) -> Box<SignalEntry> {
        SignalEntry::new(Self::IsAlive(IsAlive { sender, reference }))
    }

    #[inline]
    pub fn flush(ty: FlushType) -> Box<SignalEntry> {
        SignalEntry::new(Self::Flush(Flush { sender: None, ty }))
    }

    #[inline]
    pub fn rpc_noreply(
        sender: Pid,
        callback: RpcCallback,
        arg: *mut (),
        priority: Priority,
    ) -> Box<SignalEntry> {
        SignalEntry::new(Self::Rpc(Rpc {
            sender,
            reference: None,
            callback,
            arg,
            priority,
        }))
    }
}
impl DynSignal for Signal {
    fn sender(&self) -> Option<WeakAddress> {
        match self {
            Self::Message(sig) => sig.sender(),
            Self::Exit(sig) => sig.sender(),
            Self::ExitLink(sig) => sig.sender(),
            Self::Monitor(sig) => sig.sender(),
            Self::Demonitor(sig) => sig.sender(),
            Self::MonitorDown(sig) => sig.sender(),
            Self::Link(sig) => sig.sender(),
            Self::Unlink(sig) => sig.sender(),
            Self::UnlinkAck(sig) => sig.sender(),
            Self::GroupLeader(sig) => sig.sender(),
            Self::IsAlive(sig) => sig.sender(),
            Self::ProcessInfo(sig) => sig.sender(),
            Self::Flush(sig) => sig.sender(),
            Self::Rpc(sig) => sig.sender(),
        }
    }
}

/// Sent by a process which wants the receiver to handle all of
/// its incoming signals, flushing its signal queue.
pub struct Flush {
    pub sender: Option<WeakAddress>,
    pub ty: FlushType,
}
impl DynSignal for Flush {
    fn sender(&self) -> Option<WeakAddress> {
        self.sender.clone()
    }
}

/// Represents a message sent by `sender` to the receiving process
pub struct Message {
    pub sender: WeakAddress,
    pub message: TermFragment,
}
impl DynSignal for Message {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone())
    }
}

/// Represents an exit signal sent by `sender` to the receiving process
///
/// It is up to the receiving process to determine how to handle exits.
/// If a process has the `trap_exits` flag set, this signal will be converted
/// to a message and placed in the message queue. Otherwise, the process
/// will set its status to `Exiting`, and begin terminating itself.
pub struct Exit {
    pub sender: Option<WeakAddress>,
    /// The exit reason provided
    pub reason: TermFragment,
    /// If true, a reason of 'normal' will also kill
    /// the receiver if it is not trapping exits
    pub normal_kills: bool,
}
impl DynSignal for Exit {
    fn sender(&self) -> Option<WeakAddress> {
        self.sender.clone()
    }
}

/// Represents establishment of a new monitor for the receiving process/port.
///
/// This signal is used to inform the receiving process/port of the monitor, so that it
/// may ensure that the monitor is triggered when the process/port exits.
pub struct Monitor {
    /// This is the target end of the monitor, for use by the receiving process/port
    ///
    /// It is expected that the receiving process will add this monitor to its monitor
    /// list upon receipt. Subsequent signals relating to this monitor will only refer to
    /// it by reference, so those messages can only be handled if the receiver keeps this
    /// entry on hand.
    pub monitor: Arc<MonitorEntry>,
}
impl DynSignal for Monitor {
    fn sender(&self) -> Option<WeakAddress> {
        self.monitor.origin()
    }
}

/// Represents cancellation of the monitor identified by `monitor_ref`.
///
/// The sender of this signal must be the origin of the monitor.
pub struct Demonitor {
    pub sender: WeakAddress,
    /// This is the reference to the monitor entry which was held by the sender, whose
    /// ownership was passed to this signal. This can be used to directly obtain a reference
    /// to the entry in the origin monitor tree upon receipt.
    pub monitor: Arc<MonitorEntry>,
}
impl DynSignal for Demonitor {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone())
    }
}

/// Represents a monitor owned by the receiver being triggered by one of the following:
///
/// * The monitored process has exited
/// * The monitor is no longer valid as the connection to the node on which the monitored
/// process lives has been disrupted.
pub struct MonitorDown {
    /// The sender of this signal should always be the target process/port, but might
    /// be None if the sender is unavailable at the time the signal is generated
    pub sender: Option<WeakAddress>,
    /// The reason for the monitor going down
    pub reason: TermFragment,
    /// This is the reference to the monitor entry which was held by the sender, whose
    /// ownership was passed to this signal. This can be used to directly obtain a reference
    /// to the entry in the origin monitor tree upon receipt.
    pub monitor: Arc<MonitorEntry>,
}
impl DynSignal for MonitorDown {
    fn sender(&self) -> Option<WeakAddress> {
        self.sender.clone()
    }
}

/// Represents establishment of a new link to the receiving process.
///
/// This is used to inform the receiving process of the link, so that it
/// may ensure that the link is triggered when the process/port exits.
pub struct Link {
    /// This is the target end of the link, for use by the receiving process/port
    ///
    /// It is expected that the receiving process will add this link to its link
    /// list upon receipt. Subsequent signals relating to this link will implicitly
    /// assume that the receiver is able to look up the link by sender address
    /// (e.g. pid or port id).
    pub link: Arc<LinkEntry>,
}
impl DynSignal for Link {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.link.origin())
    }
}

/// Represents a request to remove a previously established link to the receiving process.
///
/// The sender of this signal must be the origin of the link
pub struct Unlink {
    /// This address acts both as the sender of the signal, and the origin address of the link
    pub sender: WeakAddress,
    /// The unique identifier associated with this unlink op
    pub id: NonZeroU64,
}
impl DynSignal for Unlink {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone())
    }
}

/// Represents an acknowledgment of an unlink signal previously sent by the receiving process.
///
/// In this scenario, the receiver originally sent an `Unlink` signal to `sender`, and this
/// signal is the response to that.
pub struct UnlinkAck {
    /// This address acts both as the sender of the signal, and the target address of the link
    pub sender: WeakAddress,
    /// The unique identifier associated with the original unlink op
    pub id: NonZeroU64,
}
impl DynSignal for UnlinkAck {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone())
    }
}

/// Represents a request to change the group leader of the receiving process.
///
/// If sent locally, a response message of `{Ref, true | badarg}` is sent to the sender,
/// where `Ref` is the reference given.
pub struct GroupLeader {
    /// The process enacting the change
    pub sender: Pid,
    /// The new group leader
    pub group_leader: Pid,
    /// The reference associated with this request
    pub reference: Reference,
}
impl DynSignal for GroupLeader {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone().into())
    }
}

/// Represents a liveness check by `sender` to the receiving process.
///
/// A response message of `{Ref, true | false}` is sent to the sender,
/// where `Ref` is the reference given.
pub struct IsAlive {
    pub sender: Pid,
    pub reference: Reference,
}
impl DynSignal for IsAlive {
    fn sender(&self) -> Option<WeakAddress> {
        Some(self.sender.clone().into())
    }
}

/// Represents a request for process info by `sender` to the receiving process.
///
/// A response message '{Ref, Result}' is sent to the sender when performed,
/// where `Ref` is the reference passed, and `Result` corresponds to the return value
/// of `erlang:process_info/{1,2}`.
pub struct ProcessInfo {
    /// A strong reference to the currently executing (sending) process
    ///
    /// This is `None` if the signal arrived via distribution
    pub sender: Option<Arc<Process>>,
    /// The atom representing the info type requested
    pub item: Atom,
    /// The reference used int he response message
    pub reference: Reference,
    /// If true, the message queue length is needed, so it will be calculated before
    /// fetching the process info
    pub need_msgq_len: bool,
}
impl DynSignal for ProcessInfo {
    fn sender(&self) -> Option<WeakAddress> {
        self.sender.as_ref().map(|p| p.pid().into())
    }
}

/// Represents a request to execute a function in the context of the receiver.
///
/// If a reference is given, the receiver will send a reply message of the form
/// `{Ref, Result}`, where `Ref` is the given reference, and `Result` is the result
/// produced by the rpc callback.
pub struct Rpc {
    pub sender: Pid,
    /// The reference to use in the rpc reply message
    ///
    /// If `None`, no reply is wanted and none will be sent.
    pub reference: Option<Reference>,
    /// A pointer to the function to call in the context of the receiver
    pub callback: RpcCallback,
    /// An opaque pointer for use by the callback to pass arguments/state/context
    pub arg: *mut (),
    /// The priority at which the rpc signal will be executed under
    pub priority: Priority,
}
unsafe impl Send for Rpc {}
unsafe impl Sync for Rpc {}
impl DynSignal for Rpc {
    fn sender(&self) -> Option<WeakAddress> {
        Some(WeakAddress::Process(self.sender.clone()))
    }
}

// An intrusive linked list adapter for storing boxed signal entries
intrusive_adapter!(pub SignalAdapter = Box<SignalEntry>: SignalEntry { link: LinkedListLink });

/// A type alias for the intrusive linked list type
type SignalList = LinkedList<SignalAdapter>;
/// A type alias for the intrusive linked list cursor type
type SignalCursor<'a> = intrusive_collections::linked_list::Cursor<'a, SignalAdapter>;
/// A type alias for the intrusive linked list mutable cursor type
type SignalCursorMut<'a> = intrusive_collections::linked_list::CursorMut<'a, SignalAdapter>;

/// Represents an entry in the signal queue
pub struct SignalEntry {
    link: LinkedListLink,
    pub signal: Signal,
}
impl SignalEntry {
    pub fn new(signal: Signal) -> Box<Self> {
        Box::new(Self {
            link: LinkedListLink::new(),
            signal,
        })
    }

    pub fn is_message(&self) -> bool {
        match self.signal {
            Signal::Message(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn sender(&self) -> Option<WeakAddress> {
        self.signal.sender()
    }
}

bitflags::bitflags! {
    pub struct SignalQueueFlags: u8 {
        /// The message queue stores data off-heap
        const OFF_HEAP = 1;
        /// The message queue stores data on-heap
        const ON_HEAP = 1 << 1;
        /// Process is flushing signals
        const FLUSHING = 1 << 4;
        /// Process has finished flushing signals
        const FLUSHED = 1 << 5;
    }
}
impl firefly_system::sync::Atom for SignalQueueFlags {
    type Repr = u8;

    #[inline]
    fn pack(self) -> Self::Repr {
        self.bits()
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        unsafe { SignalQueueFlags::from_bits_unchecked(raw) }
    }
}
impl firefly_system::sync::AtomLogic for SignalQueueFlags {}

/// This is a low-level queue used internally by [`InTransitQueue`] and [`SignalQueue`].
///
/// It is designed to track all kinds of signals, but distinguish between message and
/// non-message signals, to allow for efficient prioritization of signals over messages
/// without sacrificing ordering guarantees
#[derive(Default)]
struct Queue {
    signals: SignalList,
    messages: SignalList,
    len: usize,
}
impl Queue {
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// The number of buffers must be equal to the word size of the target,
/// as we use a single word as a bitset that indicates what buffers are
/// non-empty
const NUM_INQ_BUFFERS: usize = mem::size_of::<usize>() * 8;

struct InTransitQueue {
    buffers: [CachePadded<Mutex<Queue>>; NUM_INQ_BUFFERS],
    nonempty_slots: AtomicUsize,
    nonmessage_slots: AtomicUsize,
    /// The number of signals in-transit
    ///
    /// This is used to help provide statistics upon request about the size of a processes' message
    /// queue
    len: AtomicUsize,
}
impl InTransitQueue {
    #[inline]
    fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }
}
impl Default for InTransitQueue {
    fn default() -> Self {
        let mut buffers =
            MaybeUninit::<CachePadded<Mutex<Queue>>>::uninit_array::<NUM_INQ_BUFFERS>();
        for buffer in &mut buffers {
            buffer.write(CachePadded::new(Mutex::new(Queue::default())));
        }
        Self {
            buffers: unsafe { MaybeUninit::array_assume_init(buffers) },
            nonempty_slots: AtomicUsize::new(0),
            nonmessage_slots: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
        }
    }
}

/// The private, inner signal queue of a process
///
/// This is the final queue in which signals reside before they are handled by the recipient.
struct PrivateSignalQueue {
    received: Queue,
    cursor: *const SignalEntry,
    last_seen: *const SignalEntry,
}
impl PrivateSignalQueue {
    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.received.is_empty()
    }

    fn cursor(&self) -> SignalCursor<'_> {
        if unlikely(self.cursor.is_null()) {
            // Return a null cursor
            self.received.messages.cursor()
        } else {
            // We have an active cursor, so restore it
            unsafe { self.received.messages.cursor_from_ptr(self.cursor) }
        }
    }

    fn cursor_mut(&mut self) -> SignalCursorMut<'_> {
        if unlikely(self.cursor.is_null()) {
            // Return a null cursor
            self.received.messages.cursor_mut()
        } else {
            // We have an active cursor, so restore it
            unsafe { self.received.messages.cursor_mut_from_ptr(self.cursor) }
        }
    }
}

/// This type represents ownership over the `PrivateSignalQueue` of a process.
///
/// The holder of this lock is allowed to send signals directly to the private queue
/// of a process, flush its in-transit buffers, and receive messages from the queue.
pub struct SignalQueueLock<'a> {
    signals: &'a SignalQueue,
    queue: MutexGuard<'a, PrivateSignalQueue>,
}
impl<'a> SignalQueueLock<'a> {
    /// Gets an estimate on the number of message signals in the signal queue
    ///
    /// This number includes all signals, not just messages.
    pub fn len(&self) -> usize {
        self.queue.received.len + self.signals.in_transit.len()
    }

    /// Returns true if there are signals in the private queue
    pub fn has_pending_signals(&self) -> bool {
        !self.queue.received.signals.is_empty()
    }

    /// Returns true if the private queue is empty
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// This corresponds to the recv_peek and recv_next primops, called when entering a new receive
    /// block, and on each consecutive iteration of the search through the message queue.
    ///
    /// If there is no current receive cursor set, the first thing we do is set it to the start
    /// of the inner queue.
    ///
    /// If the inner queue has messages, we ensure the cursor points to the next unseen message in
    /// the queue, and return `Ok`.
    ///
    /// If the inner queue is empty, or we've reached the end of the queue, we take a moment to try
    /// and find more messages in the buffers of the in-transit queue. If no messages can be
    /// found, this function returns `Err`.
    pub fn try_receive(&mut self) -> Result<(), ()> {
        // Get a cursor to the next unseen element
        let cursor = self.queue.cursor();
        // If the cursor is null, this is our first attempt to receive, or
        // a previous attempt hit an empty mailbox. Try to get messages now,
        // backfilling from the in-transit buffers if necessary.
        //
        // Otherwise, we will have just called `next_message`, so as long as
        // the cursor has advanced, we proceed normally; however if the cursor
        // could not advance, meaning we have visited all the messages in the mailbox,
        // but none were matches, so we're waiting for more, try to get more messages
        // that may match.
        trace!(target: "process", "try_receive started with {} messages in the queue", self.queue.received.len);
        if !cursor.is_null() {
            trace!(target: "process", "resuming in-progress receive operation");
            // If we were unable to advance the cursor, try to fetch more messages
            if self.queue.cursor == self.queue.last_seen {
                trace!(target: "process", "cursor was not advanced on the last recv_next op: current={:p} last={:p}", self.queue.cursor, self.queue.last_seen);
                trace!(target: "process", "attempting to fetch more messages from in-transit queue");
                let queue = self.queue.deref_mut();
                self.signals
                    .try_flush_message_buffers(&mut queue.received)?;
                // Try to advance the cursor again
                trace!(target: "process", "flush successful, {} new messages in private queue", self.queue.received.len);
                self.next_message();
                if self.queue.cursor == self.queue.last_seen {
                    // No joy, return Err(())
                    trace!(target: "process", "incredibly, this didn't work: current={:p} last={:p}", self.queue.cursor, self.queue.last_seen);
                    Err(())
                } else {
                    // Success!
                    trace!(target: "process", "cursor was moved to next message in queue: {:p} (last was {:p})", self.queue.cursor, self.queue.last_seen);
                    Ok(())
                }
            } else {
                // the cursor is positioned at the next message to peek
                trace!(target: "process", "cursor is set to {:p} (last was {:p})", self.queue.cursor, self.queue.last_seen);
                Ok(())
            }
        } else {
            trace!(target: "process", "starting receive operation");
            // Initialize the cursor to the front of the queue
            let cursor = self.queue.received.messages.front();
            if unlikely(cursor.is_null()) {
                trace!(target: "process", "no messages in the private queue, attempting to fetch from in-transit queue");
                // No messages pending, try to fetch in-transit queue
                let queue = self.queue.deref_mut();
                self.signals
                    .try_flush_message_buffers(&mut queue.received)?;
                trace!(target: "process", "flush successful, {} new messages in private queue", queue.received.len);
                // We're guaranteed to have at least one message here
                let cursor = queue.received.messages.front().get().unwrap() as *const _;
                self.queue.cursor = cursor;
                trace!(target: "process", "cursor was set to front of queue: {:p} (last was {:p})", self.queue.cursor, self.queue.last_seen);
                Ok(())
            } else {
                trace!(target: "process", "messages found in the private queue, setting cursor to front of queue");
                Err(())
            }
        }
    }

    /// Moves the cursor to the next message
    pub fn next_message(&mut self) {
        let cursor = self.queue.cursor();
        assert!(!cursor.is_null());
        let next_cursor = cursor.peek_next();
        let cursor = unsafe { cursor.get().unwrap_unchecked() as *const _ };
        if !next_cursor.is_null() {
            trace!(target: "process", "recv_next advanced the cursor");
            let next_cursor = unsafe { next_cursor.get().unwrap_unchecked() as *const _ };
            self.queue.last_seen = cursor;
            self.queue.cursor = next_cursor;
        } else {
            self.queue.last_seen = cursor;
            trace!(target: "process", "recv_next found no more messages to peek, last_seen={:p}", self.queue.last_seen);
        }
    }

    /// Peeks at the term contained in the message pointed to by the receive cursor
    #[inline]
    pub fn peek_message<'b: 'a>(&'b self) -> Option<&'a Message> {
        if self.queue.cursor.is_null() {
            trace!(target: "process", "recv_peek found no messages in the queue");
            None
        } else {
            if self.queue.cursor == self.queue.last_seen {
                trace!(target: "process", "recv_peek has already seen the latest message in the queue");
                None
            } else {
                trace!(target: "process", "recv_peek succeeded");
                let sig = unsafe { &*self.queue.cursor };
                match sig.signal {
                    // This is the only type of signal the cursor points to
                    Signal::Message(ref msg) => Some(msg),
                    _ => unreachable!(),
                }
            }
        }
    }

    /// This corresponds to the `remove_message` primop, called when a message as matched
    /// by a receive, and is being removed from the queue.
    ///
    /// This function takes the message currently pointed to by the receive cursor, and
    /// pops it from the queue. The cursor is reset to null as part of this operation.
    ///
    /// # SAFETY
    ///
    /// This function is only safe to call from the receiving process.
    ///
    /// This function assumes that the cursor is valid (as prepared by `try_receive`), and
    /// will panic if there are no messages in the received queue.
    pub fn remove_message(&mut self) -> Message {
        let sig = self.queue.cursor_mut().remove().unwrap();
        trace!(target: "process", "recv_pop successful");
        match sig.signal {
            // This is the only type of signal the cursor points to
            Signal::Message(msg) => {
                self.queue.cursor = ptr::null();
                self.queue.last_seen = ptr::null();
                self.queue.received.len -= 1;
                msg
            }
            _ => unreachable!(),
        }
    }

    /// Resets the receive marker at the end of a receive
    pub fn end_receive(&mut self) {
        self.queue.cursor = ptr::null();
        self.queue.last_seen = ptr::null();
    }

    /// Pushes the given message to the front of the message queue
    ///
    /// # SAFETY
    ///
    /// This must only be called by the process which owns this signal queue
    /// while it is handling signals. This is intended for use cases such as
    /// converting a MonitorDown signal to a Message, while ensuring that the
    /// resulting message is handled at the same priority as its origin signal.
    pub unsafe fn push_next_message(&mut self, signal: Box<SignalEntry>) {
        self.queue.received.messages.push_front(signal);
        self.queue.received.len += 1;
    }

    /// This function is used to remove the next non-message signal from the queue.
    ///
    /// If there are no non-message signals available in the received queue, and `local_only` is
    /// `false`, this function will, much like `try_receive`, flush buffers from the in-transit
    /// queue in an effort to get more. Unlike `try_receive` however, only buffers containing
    /// non-message signals are flushed.
    ///
    /// If no non-message signals are available, this function returns `None`.
    pub fn pop_signal(&mut self, local_only: bool) -> Option<Box<SignalEntry>> {
        let signal = self.queue.received.signals.pop_front();
        if unlikely(signal.is_none()) {
            if local_only {
                None
            } else {
                let queue = self.queue.deref_mut();
                match self.signals.try_flush_signal_buffers(&mut queue.received) {
                    Ok(_) => queue.received.signals.pop_front(),
                    Err(_) => None,
                }
            }
        } else {
            signal
        }
    }

    /// Takes the next signal from the queue, pulling from the message queue if no non-message
    /// signals are present
    pub fn pop(&mut self) -> Option<Box<SignalEntry>> {
        match self.queue.received.signals.pop_front() {
            None => self.queue.received.messages.pop_front(),
            sig @ Some(_) => sig,
        }
    }

    /// Called when a holder of the signal queue lock wishes to push a message directly
    /// in to the private queue of a process. This will force flush the in-transit buffers
    /// to ensure signal order, as long as the signal queue is off-heap.
    pub(super) fn push_private(&mut self, signal: Box<SignalEntry>) -> SendResult {
        if self.signals.flags().contains(SignalQueueFlags::OFF_HEAP) {
            let nonempty_slots = self
                .signals
                .in_transit
                .nonempty_slots
                .swap(0, Ordering::Relaxed);
            let queue = self.queue.deref_mut();
            self.signals
                .try_flush_buffers(nonempty_slots, &mut queue.received)
                .ok();
        }

        let is_message = signal.is_message();
        let mut result = SendResult::SUCCESS;

        if is_message {
            result |= SendResult::MESSAGE;
            self.queue.received.messages.push_back(signal);
        } else {
            let set_signals_in_queue = self.queue.received.signals.is_empty();
            self.queue.received.signals.push_back(signal);
            if set_signals_in_queue {
                result |= SendResult::NOTIFY_IN_TRANSIT;
            }
        }
        // Record a message in the in-transit queue for statistics
        //
        // We only really want to count messages, but we only need an estimate
        // and by tracking both signals and messages together, we can easily manage
        // the counter without having to count every message in each buffer when we
        // flush
        self.queue.received.len += 1;

        result
    }

    /// Performs a complete flush of the in-transit buffers to the private queue
    pub fn flush_buffers(&mut self) {
        let nonempty_slots = self
            .signals
            .in_transit
            .nonempty_slots
            .swap(0, Ordering::Relaxed);
        let queue = self.queue.deref_mut();
        self.signals
            .try_flush_buffers(nonempty_slots, &mut queue.received)
            .ok();
    }
}
impl<'a> Deref for SignalQueueLock<'a> {
    type Target = SignalQueue;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.signals
    }
}

bitflags::bitflags! {
    pub struct SendResult: u8 {
        /// Signal was rejected because the recipient is exiting
        const REJECTED = 0;
        /// Signal was successfully sent
        const SUCCESS = 1 << 1;
        /// Signal sent was a message
        const MESSAGE = 1 << 2;
        /// Signal was pushed in to an empty in-transit queue
        ///
        /// The process should have its HAS_IN_TRANSIT_SIGNALS flag set
        const NOTIFY_IN_TRANSIT = 1 << 3;
    }
}

pub struct SignalQueue {
    flags: Atomic<SignalQueueFlags>,
    in_transit: InTransitQueue,
    private: Mutex<PrivateSignalQueue>,
}
impl Default for SignalQueue {
    fn default() -> Self {
        Self {
            flags: Atomic::new(SignalQueueFlags::OFF_HEAP),
            in_transit: InTransitQueue::default(),
            private: Mutex::new(PrivateSignalQueue {
                received: Queue::default(),
                cursor: ptr::null(),
                last_seen: ptr::null(),
            }),
        }
    }
}
impl SignalQueue {
    /// Acquires the signal queue lock for use by the caller
    pub fn lock<'a>(&'a self) -> SignalQueueLock<'a> {
        let queue = self.private.lock();
        SignalQueueLock {
            signals: self,
            queue,
        }
    }

    /// Returns the current flags set on this queue
    pub fn flags(&self) -> SignalQueueFlags {
        self.flags.load(Ordering::Release)
    }

    /// Sets one or more flags on this queue, returning the previous flags
    pub fn set_flags(&self, flags: SignalQueueFlags) -> SignalQueueFlags {
        self.flags.fetch_or(flags, Ordering::Acquire)
    }

    /// Sets one or more flags on this queue, returning the previous flags
    pub fn remove_flags(&self, flags: SignalQueueFlags) -> SignalQueueFlags {
        self.flags.fetch_and(!flags, Ordering::Acquire)
    }

    /// Called from the context of the sending entity
    ///
    /// This function places the given signal in the in-transit queue of the receiver
    ///
    /// Returns `Ok(true)` if successful and the signal was a message.
    /// Returns `Ok(false)` if successful and the signal was _not_ a message.
    /// Returs `Err` if the signal is rejected by the receiving process (i.e. it is exiting)
    pub(super) fn push(&self, signal: Box<SignalEntry>) -> SendResult {
        let is_message = signal.is_message();
        let sender = signal.sender().unwrap_or(WeakAddress::System);
        let slot = hash_address_to_index(&sender);
        let mut result = SendResult::SUCCESS;

        let mut buffer = self.in_transit.buffers[slot].lock();
        if buffer.is_empty() {
            // The buffer is empty so we need to notify the receiver,
            // unless some other slot is non-empty, in which case another
            // enqueuer has already (or will) notify the receiver.
            self.in_transit
                .nonempty_slots
                .fetch_or(1 << slot, Ordering::Relaxed);
        }

        let nonmessage_slots_before;
        if !is_message {
            if buffer.signals.is_empty() {
                // We're inserting a non-message signal and no non-message signals were in the
                // buffer before. This means we have to update the non-message
                // status of this buffer.
                //
                // Acquire ordering is used to avoid reordering with a store when enqueing signals
                // below
                nonmessage_slots_before = self
                    .in_transit
                    .nonmessage_slots
                    .fetch_or(1 << slot, Ordering::Acquire);
            } else {
                nonmessage_slots_before = 1;
            }
        } else {
            nonmessage_slots_before = 1;
        }

        if is_message {
            result |= SendResult::MESSAGE;
            buffer.messages.push_back(signal);
        } else {
            let set_signals_in_queue = nonmessage_slots_before == 0;
            buffer.signals.push_back(signal);
            if set_signals_in_queue {
                result |= SendResult::NOTIFY_IN_TRANSIT;
            }
        }
        // Record a message in the in-transit queue for statistics
        //
        // We only really want to count messages, but we only need an estimate
        // and by tracking both signals and messages together, we can easily manage
        // the counter without having to count every message in each buffer when we
        // flush
        self.in_transit.len.fetch_add(1, Ordering::Relaxed);
        buffer.len += 1;

        // Release the buffer lock
        drop(buffer);

        result
    }

    /// This function is called when attempting to fetch any pending messages and there
    /// aren't any in the received queue. When this occurs, we flush all non-empty buffers
    /// containing messages.
    ///
    /// This function returns `Ok` if there are messages available in the received
    /// queue after flushing. Otherwise, `Err`.
    fn try_flush_message_buffers(&self, rq: &mut Queue) -> Result<(), ()> {
        let nonempty_slots = self.in_transit.nonempty_slots.swap(0, Ordering::Relaxed);
        self.try_flush_buffers(nonempty_slots, rq)?;
        // We only return `Ok` if the messages queue is non-empty
        if rq.messages.is_empty() {
            Err(())
        } else {
            Ok(())
        }
    }

    /// This function is called when attempting to fetch any pending signals and there
    /// aren't any in the received queue, so we check if there are any in the in-transit
    /// buffers and flush buffers containing pending signals.
    ///
    /// This function returns `Ok` if there are pending signals available in the received
    /// queue after flushing. Otherwise, `Err`.
    fn try_flush_signal_buffers(&self, rq: &mut Queue) -> Result<(), ()> {
        let nonempty_slots = self.in_transit.nonmessage_slots.swap(0, Ordering::Relaxed);
        self.try_flush_buffers(nonempty_slots, rq)?;
        // We only return `Ok` if the signals queue is non-empty
        if rq.signals.is_empty() {
            Err(())
        } else {
            Ok(())
        }
    }

    /// This function is called when attempting to fetch more signals from the in-transit
    /// buffers, due to an empty received queue. Because we distinguish between messages and
    /// signals, functions which call this care about different sets of buffers, as not all
    /// non-empty buffers may contain signals of the desired type.
    ///
    /// To facilitate this, the caller provides a bitset containing the non-empty slots to
    /// be flushed, if there are any available. If there are no non-empty slots to flush, this
    /// function returns `Err`, otherwise it returns `Ok` when the buffers have been flushed to
    /// the received queue.
    fn try_flush_buffers(&self, mut nonempty_slots: usize, rq: &mut Queue) -> Result<(), ()> {
        // If there are no non-empty buffers, we have nothing to do
        if nonempty_slots == 0 {
            return Err(());
        }

        // For each non-empty buffer, acquire the buffer lock and transfer its contents
        // to the end of the provided queue
        //
        // Start by skipping any slots which are empty
        let mut shift = nonempty_slots.trailing_zeros();
        let mut slot = shift as usize;
        loop {
            // We're done when we have no more non-empty buffers to check
            if slot == NUM_INQ_BUFFERS {
                break;
            }

            let len;
            let signals;
            let messages;
            // Acquire the buffer lock just long enough to take the contents out
            {
                let mut buffer = self.in_transit.buffers[slot].lock();
                debug_assert!(!buffer.is_empty());
                len = buffer.len;
                signals = buffer.signals.take();
                messages = buffer.messages.take();
                buffer.len = 0;
                self.in_transit.len.fetch_sub(len, Ordering::Relaxed);
            }

            // Then handle the remaining work after the lock is released
            rq.signals.back_mut().splice_after(signals);
            rq.messages.back_mut().splice_after(messages);
            rq.len += len;

            // Calculate the slot containing the next non-empty buffer
            nonempty_slots = nonempty_slots >> (shift + 1);
            shift = nonempty_slots.trailing_zeros();
            slot = shift as usize;
        }

        Ok(())
    }
}

/// Computes the in-transit queue buffer index for a signal from the given address
fn hash_address_to_index(address: &WeakAddress) -> usize {
    let mut hasher = rustc_hash::FxHasher::default();
    address.hash(&mut hasher);
    let hash = hasher.finish();
    let index = hash % (NUM_INQ_BUFFERS as u64);
    index as usize
}

use std::convert::TryInto;
use std::sync::Arc;

use liblumen_alloc::erts::message::Message;
use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::timeout::{ReceiveTimeout, Timeout};

use lumen_rt_core::process::current_process;
use lumen_rt_core::time::monotonic;
use lumen_rt_core::timer::{self, SourceEvent};

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReceiveState {
    // Indicates to the caller that a message is available to peek
    Peek = 0,
    // Indicates to the caller that no message was available and the process
    // was transitioned to the waiting state.
    Wait = 1,
    // Indicates to the caller that the receive timed out
    Timeout = 2,
}

/// This structure manages the context for a receive state machine.
///
/// It is critical that the layout for this structure be carefully
/// maintained, as the compiler generates code which accesses it. Any
/// changes that modify the layout or semantics of this structure must
/// be synchronized with codegen.
#[repr(C)]
pub struct ReceiveContext {
    /// This field contains the absolute timeout in monotonic time (if applicable)
    /// associated with this receive operation.
    timeout: ReceiveTimeout,
    /// This field contains a timer reference used to wake the process if a message
    /// is never received, if a timeout was set. If no timeout was set, then this
    /// has a value of Term::None, and is unused.
    timer_reference: Term,
    /// This field is a pointer to a Message stored in the mailbox. Each Message is
    /// an entry in an intrusive, doubly-linked list which forms the queue of the
    /// mailbox. This pointer is used to get a cursor in that list, which is then
    /// used to obtain a reference to the message if the cursor is valid and points
    /// to a message which is still in the mailbox. If the message is not in the mailbox,
    /// the cursor returns None, otherwise it returns Some(&Message).
    ///
    /// NOTE: The only valid values for this pointer are null, or a pointer to a Message struct
    /// which is still live at the given address. For this reason it is not permitted to
    /// perform garbage collection while a receive context is live. This is in general not an
    /// issue, since a GC should only occur prior to, or after, a receive context is released.
    /// However, as an additional layer of defense, we add an assertion in the garbage_collect function
    /// of the process that raises if the process is in the Waiting state (the only state in which it
    /// should be possible for the garbage collector to be invoked).
    ///
    /// In the future, if we need to permit garbage collection during receives, we can do this
    /// by storing the live receive context in the mailbox state, and fixing up this pointer during
    /// GC. Rather than getting the whole struct when calling `start`, we'd just return a pointer to
    /// the context instead.
    message: *const Message,
}
impl ReceiveContext {
    #[inline]
    fn new(arc_process: Arc<Process>, timeout: Timeout) -> Self {
        let now = monotonic::time();
        let timeout = ReceiveTimeout::new(now, timeout);
        let timer_reference = if let Some(monotonic) = timeout.monotonic() {
            timer::start(monotonic, SourceEvent::StopWaiting, arc_process).unwrap()
        } else {
            Term::NONE
        };

        Self {
            timer_reference,
            message: core::ptr::null(),
            timeout,
        }
    }

    #[inline]
    fn timeout(&mut self) -> bool {
        let now = monotonic::time();
        if self.timeout.is_timed_out(now) {
            self.cancel_timer();
            true
        } else {
            false
        }
    }

    fn cancel_timer(&mut self) {
        if self.timer_reference != Term::NONE {
            let boxed_timer_reference: Boxed<Reference> = self.timer_reference.try_into().unwrap();
            timer::cancel(&boxed_timer_reference);
            self.timer_reference = Term::NONE;
        }
    }
}

/// This function is called to construct a receive state machine, it performs no other work beyond just
/// setting up the context for the state machine itself. The following state transition graph summarizes how the
/// caller is expected to manage the state machine.
///
/// start -> next <---------<-------- yield
///           |             |           |
///           |             |           |
///           v             |           |
///           received --> peek         |
///           |             |           |
///           |             v           |
///           |            pop          |
///           |             |           |
///           |             v           |
///           timeout ---> done         |
///           |                         |
///           wait ---------------------^
///
/// The state transitions here are `start`, `next`, `peek`, `pop` and `done`, and generally correspond to functions
/// in this module. The `wait` state occurs when there are no messages available, and so the process is
/// suspended until either a message is received or a timeout occurs. In either case, the process will resume,
/// try to transition via `next` and branch accordingly.
///
/// Yielding is implemented by the caller, the only behavioral semantics that must be preserved are that
/// garbage collection may not occur during this yield, and that other processes must have the opportunity
/// to execute during the yield so as to ensure forward progress.
///
/// When `next` transitions to the received state, the next transition is `peek`, which will either `pop`
/// the received message in the case where the peeked message matches one of the receive patterns; or it
/// will transition to `next` and try again with the next message.
///
/// NOTE: As of this writing, `peek` is implemented via generated code that directly accesses the message data.
#[export_name = "__lumen_builtin_receive_start"]
pub extern "C-unwind" fn builtin_receive_start(timeout: Term) -> ReceiveContext {
    let to = match timeout.decode().unwrap() {
        TypedTerm::Atom(atom) if atom == "infinity" => Timeout::Infinity,
        TypedTerm::SmallInteger(si) => Timeout::from_millis(si).expect("invalid timeout value"),
        _ => unreachable!("should never get non-atom/non-integer receive timeout"),
    };
    let p = current_process();
    ReceiveContext::new(p.clone(), to)
}

/// This function is called in three places:
///
/// * Upon entering the receive state machine
/// * During a selective receive, when a peeked message does not match
/// * After waiting to be woken by the scheduler due to message receipt or time out
///
/// This function determines whether or not a message is available, and what state to transition
/// to next.
#[export_name = "__lumen_builtin_receive_next"]
pub extern "C-unwind" fn builtin_receive_next(context: &mut ReceiveContext) -> ReceiveState {
    let p = current_process();
    let mbox_lock = p.mailbox.lock();
    let mut mbox = mbox_lock.borrow_mut();

    let cursor = if context.message.is_null() {
        // If no cursor has been obtained yet, try again
        mbox.cursor()
    } else {
        // If we have a cursor, move it to the next message in the queue
        let mut cursor = unsafe { mbox.cursor_from_ptr(context.message) };
        cursor.move_prev();
        cursor
    };

    // If we have a message, proceed to `peek`
    //
    if let Some(message) = cursor.get() {
        context.message = message as *const Message;
        ReceiveState::Peek
    } else if context.timeout() {
        ReceiveState::Timeout
    } else {
        p.wait();
        ReceiveState::Wait
    }
}

/// This function is called after a message was peeked, matched successfully and the receive
/// state machine is entering its exit phase. The peeked message is removed from the mailbox,
/// but its storage is left as-is, i.e. messages allocated in heap fragments remain in their
/// fragment until a garbage collection is performed. Since a GC cycle fixes up any pointers
/// contained in roots or the heap, we don't need to concern ourselves with where the terms
/// live at this stage.
#[export_name = "__lumen_builtin_receive_pop"]
pub extern "C-unwind" fn builtin_receive_pop(context: &mut ReceiveContext) {
    let p = current_process();
    let mbox_lock = p.mailbox.lock();
    let mut mbox = mbox_lock.borrow_mut();
    // Remove the message at the current cursor
    mbox.remove(context.message);
    // Reset the cursor state in the receive context
    context.message = core::ptr::null();
}

/// This function is called when the receive state machine is exiting and is used to clean
/// up any data in the context which we don't want to leave dangling. For now, that is simply
/// the timer associated with the context.
#[export_name = "__lumen_builtin_receive_done"]
pub extern "C-unwind" fn builtin_receive_done(context: &mut ReceiveContext) {
    context.cancel_timer();
}

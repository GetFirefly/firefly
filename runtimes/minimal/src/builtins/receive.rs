use std::convert::TryInto;
use std::panic;
use std::ptr;
use std::sync::Arc;

use liblumen_alloc::erts::process::Process;
use liblumen_alloc::erts::term::prelude::*;
use liblumen_alloc::erts::timeout::{ReceiveTimeout, Timeout};

use lumen_rt_core::process::current_process;
use lumen_rt_core::time::monotonic;
use lumen_rt_core::timer::{self, SourceEvent};

extern "C" {
    #[link_name = "__lumen_builtin_yield"]
    fn builtin_yield() -> bool;
}

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ReceiveState {
    // Indicates to the caller that an unrecoverable error occurred
    Error = 0,
    // Used to indicate initialized state, prior to first attempt to receive
    Ready = 1,
    // Indicates to the caller that a message was received
    Received = 2,
    // Indicates to the caller that the receive timed out
    Timeout = 3,
}

/// This structure manages the context for a single receive operation,
/// it is created by `receive_start`, modified during `receive_wait`,
/// and used during cleanup in `receive_done` to determine what, if any,
/// cleanup needs to be performed.
///
/// It is read from non-Rust code, so we use `repr(C)`.
#[repr(C)]
pub struct ReceiveContext {
    timeout: ReceiveTimeout,
    message: Term,
    timer_reference: Term,
    state: ReceiveState,
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
            state: ReceiveState::Ready,
            message: Term::NONE,
            timer_reference,
            timeout,
        }
    }

    #[inline]
    fn with_message(&mut self, message: Term) {
        self.state = ReceiveState::Received;
        self.message = message;
    }

    #[inline]
    fn with_timeout(&mut self) {
        self.state = ReceiveState::Timeout;
        self.message = Term::NONE;
    }

    #[inline]
    fn should_time_out(&self) -> bool {
        let now = monotonic::time();
        self.timeout.is_timed_out(now)
    }

    fn cancel_timer(&mut self) {
        if self.timer_reference != Term::NONE {
            let boxed_timer_reference: Boxed<Reference> = self.timer_reference.try_into().unwrap();
            timer::cancel(&boxed_timer_reference);
            self.timer_reference = Term::NONE;
        }
    }
}

#[export_name = "__lumen_builtin_receive_start"]
pub extern "C" fn builtin_receive_start(timeout: Term) -> *mut ReceiveContext {
    let result = panic::catch_unwind(move || {
        let to = match timeout.decode().unwrap() {
            TypedTerm::Atom(atom) if atom == "infinity" => Timeout::Infinity,
            TypedTerm::SmallInteger(si) => Timeout::from_millis(si).expect("invalid timeout value"),
            _ => unreachable!("should never get non-atom/non-integer receive timeout"),
        };
        // TODO: It would be best if ReceiveContext was repr(C) so we
        // could keep it on the stack rather than heap allocate here
        let p = current_process();
        let context = Box::new(ReceiveContext::new(p.clone(), to));
        let mbox = p.mailbox.lock();
        mbox.borrow().recv_start();
        Box::into_raw(context)
    });
    if let Ok(res) = result {
        res
    } else {
        ptr::null_mut()
    }
}

#[export_name = "__lumen_builtin_receive_wait"]
pub extern "C" fn builtin_receive_wait(ctx: *mut ReceiveContext) -> ReceiveState {
    let result = panic::catch_unwind(move || {
        let context = unsafe { &mut *ctx };
        loop {
            {
                let p = current_process();
                let mbox_lock = p.mailbox.lock();
                let mut mbox = mbox_lock.borrow_mut();
                if let Some(msg) = mbox.recv_peek() {
                    mbox.recv_increment();
                    context.with_message(msg);
                    break ReceiveState::Received;
                } else if context.should_time_out() {
                    context.with_timeout();
                    break ReceiveState::Timeout;
                } else {
                    p.wait();
                }
            }
            // We put our yield here to ensure that we're not holding
            // the mailbox lock while waiting, when resuming from the
            // yield, we'll continue looping
            unsafe {
                builtin_yield();
            }
        }
    });
    if let Ok(res) = result {
        res
    } else {
        ReceiveState::Error
    }
}

#[export_name = "__lumen_builtin_receive_message"]
pub extern "C" fn builtin_receive_message(ctx: *mut ReceiveContext) -> Term {
    let context = unsafe { &*ctx };
    context.message
}

#[export_name = "__lumen_builtin_receive_done"]
pub extern "C" fn builtin_receive_done(ctx: *mut ReceiveContext) -> bool {
    let mut context = unsafe { Box::from_raw(ctx) };
    let result = panic::catch_unwind(|| {
        let p = current_process();
        let mbox_lock = p.mailbox.lock();
        let mut mbox = mbox_lock.borrow_mut();

        context.cancel_timer();

        match context.state {
            ReceiveState::Received => {
                mbox.recv_received();
            }
            ReceiveState::Timeout => {
                mbox.recv_timeout();
            }
            _ => {
                unreachable!();
            }
        }

        true
    });
    result.is_ok()
}

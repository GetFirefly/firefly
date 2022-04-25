use crate::erts::exception::ErlangException;
use crate::erts::term::prelude::Term;

/// This type is used to communicate error information between
/// the native code of a process, and the scheduler/caller.
///
/// Error info (when applicable) is communicated separately.
#[allow(unused)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ProcessSignal {
    /// No signal set
    None = 0,
    /// The process should yield/has yielded
    Yield,
    /// Operation failed due to allocation failure,
    /// or process requires garbage collection
    GarbageCollect,
    /// The process raised an error
    Error,
    /// The process exited
    Exit,
}

/// This struct is used to represent the multi-value return calling convention used by the compiler.
///
/// When a call is successful, `exception` is always null, and `value` is always a valid term.
///
/// When a call fails, `exception` is always non-null pointer to the current process exception, and
/// `value` is `Term::NONE`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ErlangResult {
    pub value: Term,
    pub exception: *mut ErlangException,
}
impl ErlangResult {
    #[inline]
    pub fn ok(value: Term) -> Self {
        Self {
            value,
            exception: core::ptr::null_mut(),
        }
    }

    #[inline]
    pub fn error(exception: *mut ErlangException) -> Self {
        Self {
            value: Term::NONE,
            exception,
        }
    }
}

extern "C" {
    #[link_name = "__lumen_process_signal"]
    #[thread_local]
    static mut PROCESS_SIGNAL: ProcessSignal;
}

/// Returns the current value of the process signal
#[inline(always)]
pub fn process_signal() -> ProcessSignal {
    unsafe { PROCESS_SIGNAL }
}

/// Sets the process signal value
#[inline(always)]
pub fn set_process_signal(value: ProcessSignal) {
    unsafe {
        PROCESS_SIGNAL = value;
    }
}

/// Clears the process signal value
#[inline(always)]
pub fn clear_process_signal() {
    unsafe {
        PROCESS_SIGNAL = ProcessSignal::None;
    }
}

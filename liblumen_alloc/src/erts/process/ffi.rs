use core::alloc::Layout;

use crate::erts::term::arch::Repr;
use crate::erts::term::prelude::{Encoded, Term};

use super::Process;

macro_rules! abort_with_message {
    ($message:expr) => {{
        println!($message);
        std::process::abort();
    }};
    ($format:expr, $($arg:expr),+) => {{
        println!($format, $($arg),+);
        std::process::abort();
    }}
}

/// This type is used to communicate error information between
/// the native code of a process, and the scheduler/caller.
///
/// Error info (when applicable) is communicated separately.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ProcessSignal {
    /// No signal set
    None = 0,
    /// The process should yield/has yielded
    Yield,
    /// The process raised an error
    Error,
    /// The process exited
    Exit,
}

extern "C" {
    #[link_name = "__lumen_proc"]
    #[thread_local]
    static mut CURRENT_PROCESS: *mut ();

    #[link_name = "__lumen_proc_reductions"]
    #[thread_local]
    static mut PROCESS_REDUCTIONS: u32;

    #[link_name = "__lumen_proc_signal"]
    #[thread_local]
    static mut PROCESS_SIGNAL: ProcessSignal;

    #[link_name = "__lumen_proc_error"]
    #[thread_local]
    static PROCESS_ERROR: Term;

    #[link_name = "__lumen_proc_sp"]
    #[thread_local]
    static PROCESS_STACK_POINTER: *mut u8;

    #[link_name = "__lumen_proc_fp"]
    #[thread_local]
    static PROCESS_FRAME_POINTER: *mut u8;
}

/// Returns the current value of the process signal
pub fn process_signal() -> ProcessSignal {
    unsafe { PROCESS_SIGNAL }
}

/// Sets the process signal value
pub fn set_process_signal(value: ProcessSignal) {
    unsafe {
        PROCESS_SIGNAL = value;
    }
}

/// Clears the process signal value
pub fn clear_process_signal() {
    unsafe {
        PROCESS_SIGNAL = ProcessSignal::None;
    }
}

/// Returns the current reduction count
pub fn process_reductions() -> u32 {
    unsafe { PROCESS_REDUCTIONS }
}

/// Sets the current reduction count
pub fn set_process_reductions(value: u32) {
    unsafe {
        PROCESS_REDUCTIONS = value;
    }
}

/// Gets the process error value
pub fn process_error() -> Option<Term> {
    let err = unsafe { PROCESS_ERROR };
    if err.is_none() || err.as_usize() == 0 {
        None
    } else {
        Some(err)
    }
}

/// Used to allocate memory on the process stack, rather than the native stack
#[link_name = "__lumen_builtin_alloca"]
extern "C" fn alloca(size: usize, align: usize) -> Option<*mut u8> {
    let proc = unsafe { current_process() };
    let layout = Layout::from_size_align(size, align).unwrap_or_else(|_| {
        abort_with_message!(
            "__lumen_builtin_alloca got invalid layout (size={}, align={})",
            size,
            align
        )
    });
    unsafe {
        proc.alloca_layout(layout)
            .ok()
            .map(|nn| nn.as_ptr() as *mut u8)
    }
}

/// Used to allocate space for a stack frame. These frames are opaque to Rust,
/// but are used to store the metadata needed to resume execution after suspension
#[link_name = "__lumen_builtin_allocate_frame"]
extern "C" fn alloc_frame(_size: usize, _align: usize) -> Option<*mut u8> {
    todo!()
}

/// Used to allocate space on the process heap directly from native code
#[link_name = "__lumen_builtin_malloc"]
extern "C" fn malloc(size: usize, align: usize) -> Option<*mut u8> {
    let proc = unsafe { current_process() };
    let layout = Layout::from_size_align(size, align).unwrap_or_else(|_| {
        abort_with_message!(
            "__lumen_builtin_alloca got invalid layout (size={}, align={})",
            size,
            align
        )
    });
    unsafe {
        proc.alloc_nofrag_layout(layout)
            .ok()
            .map(|nn| nn.as_ptr() as *mut u8)
    }
}

#[inline]
unsafe fn current_process<'a>() -> &'a Process {
    let proc = CURRENT_PROCESS as *mut Process;
    if proc.is_null() {
        abort_with_message!("__lumen_builtin_alloca called with no process!");
    } else {
        &*proc
    }
}

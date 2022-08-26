use std::sync::Arc;

use firefly_profiling::{SelfProfiler, StringId, TimingGuard};

use crate::support::StringRef;

/// This struct links our profiler subsystem to before/after pass callback-driven
/// hooks provided by LLVM. In essence, when each pass starts, we push a new
/// timing guard on the stack, and when each pass terminates, we pop it off the
/// stack and drop it.
#[repr(C)]
pub struct LlvmSelfProfiler<'a> {
    profiler: Arc<SelfProfiler>,
    stack: Vec<TimingGuard<'a>>,
    llvm_pass_event_kind: StringId,
}

impl<'a> LlvmSelfProfiler<'a> {
    pub fn new(profiler: Arc<SelfProfiler>) -> Self {
        let llvm_pass_event_kind = profiler.get_or_alloc_cached_string("LLVM Pass");
        Self {
            profiler,
            stack: Vec::default(),
            llvm_pass_event_kind,
        }
    }

    fn before_pass_callback(&'a mut self, event_id: StringId) {
        self.stack.push(TimingGuard::start(
            &self.profiler,
            self.llvm_pass_event_kind,
            event_id,
        ));
    }

    fn after_pass_callback(&mut self) {
        self.stack.pop();
    }
}

pub(crate) unsafe extern "C" fn selfprofile_before_pass_callback(
    profiler: *mut std::ffi::c_void,
    pass_name: *const std::os::raw::c_char,
    ir_name: *const std::os::raw::c_char,
) {
    let profiler = &mut *(profiler as *mut LlvmSelfProfiler<'_>);
    let pass_name = StringRef::from_ptr(pass_name);
    let ir_name = StringRef::from_ptr(ir_name);
    let event_label = profiler
        .profiler
        .get_or_alloc_cached_string(format!("{}: {}", &pass_name, &ir_name));
    profiler.before_pass_callback(event_label);
}

pub(crate) unsafe extern "C" fn selfprofile_after_pass_callback(profiler: *mut std::ffi::c_void) {
    let profiler = &mut *(profiler as *mut LlvmSelfProfiler<'_>);
    profiler.after_pass_callback();
}

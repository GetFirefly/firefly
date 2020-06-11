use std::ffi::CStr;
use std::sync::Arc;

use liblumen_profiling::{SelfProfiler, StringId, TimingGuard};

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

    fn before_pass_callback(&'a mut self, pass_name: &str, ir_name: &str) {
        let event_id = llvm_args_to_string_id(&self.profiler, pass_name, ir_name);

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

pub unsafe extern "C" fn selfprofile_before_pass_callback(
    llvm_self_profiler: *mut std::ffi::c_void,
    pass_name: *const std::os::raw::c_char,
    ir_name: *const std::os::raw::c_char,
) {
    let llvm_self_profiler = &mut *(llvm_self_profiler as *mut LlvmSelfProfiler<'_>);
    let pass_name = CStr::from_ptr(pass_name).to_str().expect("valid UTF-8");
    let ir_name = CStr::from_ptr(ir_name).to_str().expect("valid UTF-8");
    llvm_self_profiler.before_pass_callback(pass_name, ir_name);
}

pub unsafe extern "C" fn selfprofile_after_pass_callback(
    llvm_self_profiler: *mut std::ffi::c_void,
) {
    let llvm_self_profiler = &mut *(llvm_self_profiler as *mut LlvmSelfProfiler<'_>);
    llvm_self_profiler.after_pass_callback();
}

#[inline]
fn llvm_args_to_string_id(profiler: &SelfProfiler, pass_name: &str, ir_name: &str) -> StringId {
    let event_label = format!("{}: {}", pass_name, ir_name);
    profiler.get_or_alloc_cached_string(event_label)
}

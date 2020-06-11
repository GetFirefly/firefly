use std::ptr;

use anyhow::anyhow;

use liblumen_profiling::SelfProfilerRef;

use crate::enums::{self, CodeGenOptLevel, CodeGenOptSize};
use crate::profiling::{self, LlvmSelfProfiler};
use crate::target::{TargetMachine, TargetMachineRef};
use crate::{Module, ModuleRef};

mod ffi {
    use super::*;

    extern "C" {
        pub fn LLVMLumenInitializePasses();
        pub fn LLVMLumenPrintPasses();
        pub fn LLVMLumenOptimize(
            module: ModuleRef,
            target_machine: TargetMachineRef,
            config: &OptimizerConfig,
            error_out: *mut *const libc::c_char,
        ) -> bool;
    }
}

/// Initializes all LLVM/MLIR passes
pub fn init() {
    unsafe {
        ffi::LLVMLumenInitializePasses();
    }
}

/// Prints all of the currently available LLVM/MLIR passes
///
/// NOTE: Can be called without initializing LLVM
pub fn print() {
    unsafe {
        ffi::LLVMLumenPrintPasses();
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum PassBuilderOptLevel {
    O0 = 0,
    O1,
    O2,
    O3,
    Os,
    Oz,
}
impl PassBuilderOptLevel {
    pub fn from_codegen_opts(speed: CodeGenOptLevel, size: CodeGenOptSize) -> Self {
        match size {
            CodeGenOptSize::Default if speed > CodeGenOptLevel::None => Self::Os,
            CodeGenOptSize::Aggressive if speed > CodeGenOptLevel::None => Self::Oz,
            _ => match speed {
                CodeGenOptLevel::Less => Self::O1,
                CodeGenOptLevel::Default => Self::O2,
                CodeGenOptLevel::Aggressive => Self::O3,
                _ => Self::O0,
            },
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum OptStage {
    PreLinkNoLTO,
    PreLinkThinLTO,
    PreLinkFatLTO,
    ThinLTO,
    FatLTO,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct SanitizerOptions {
    memory: bool,
    thread: bool,
    address: bool,
    recover: bool,
    memory_track_origins: libc::c_uint,
}
impl Default for SanitizerOptions {
    fn default() -> Self {
        Self {
            memory: false,
            thread: false,
            address: false,
            recover: false,
            memory_track_origins: 0,
        }
    }
}

pub type SelfProfileBeforePassCallback =
    unsafe extern "C" fn(*mut libc::c_void, *const libc::c_char, *const libc::c_char);
pub type SelfProfileAfterPassCallback = unsafe extern "C" fn(*mut libc::c_void);

#[repr(C)]
#[derive(Debug)]
pub struct OptimizerConfig {
    pipeline: *const libc::c_char,
    opt_level: PassBuilderOptLevel,
    opt_stage: OptStage,
    sanitizer_opts: SanitizerOptions,
    debug: bool,
    verify: bool,
    use_thinlto_buffers: bool,
    disable_simplify_lib_calls: bool,
    emit_summary_index: bool,
    emit_module_hash: bool,
    preserve_use_list_order: bool,
    profiler: *mut libc::c_void,
    before_pass: SelfProfileBeforePassCallback,
    after_pass: SelfProfileAfterPassCallback,
}
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            pipeline: ptr::null(),
            opt_level: PassBuilderOptLevel::O0,
            opt_stage: OptStage::PreLinkNoLTO,
            sanitizer_opts: Default::default(),
            debug: false,
            verify: false,
            use_thinlto_buffers: false,
            disable_simplify_lib_calls: false,
            emit_summary_index: false,
            emit_module_hash: false,
            preserve_use_list_order: false,
            profiler: ptr::null_mut(),
            before_pass: profiling::selfprofile_before_pass_callback,
            after_pass: profiling::selfprofile_after_pass_callback,
        }
    }
}

pub struct PassManager {
    config: OptimizerConfig,
}
impl PassManager {
    pub fn new() -> Self {
        Self {
            config: Default::default(),
        }
    }

    pub fn debug(&mut self, debug: bool) {
        self.config.debug = debug;
    }

    pub fn verify(&mut self, verify: bool) {
        self.config.verify = verify;
    }

    pub fn optimize(&mut self, level: PassBuilderOptLevel) {
        self.config.opt_level = level;
    }

    pub fn stage(&mut self, stage: OptStage) {
        self.config.opt_stage = stage;
    }

    pub fn sanitize_memory(&mut self, track_origins: u32) {
        self.config.sanitizer_opts.memory = true;
        self.config.sanitizer_opts.memory_track_origins = track_origins;
    }

    pub fn sanitize_thread(&mut self) {
        self.config.sanitizer_opts.thread = true;
    }

    pub fn sanitize_address(&mut self) {
        self.config.sanitizer_opts.address = true;
    }

    pub fn profile(&mut self, profiler: &SelfProfilerRef) {
        self.config.profiler = if profiler.llvm_recording_enabled() {
            let mut llvm_profiler = LlvmSelfProfiler::new(profiler.get_self_profiler().unwrap());
            &mut llvm_profiler as *mut _ as *mut libc::c_void
        } else {
            std::ptr::null_mut()
        };
    }

    pub fn run(self, module: &mut Module, target_machine: &TargetMachine) -> anyhow::Result<()> {
        use std::ffi::CStr;
        use std::mem::MaybeUninit;

        let mut error = MaybeUninit::<*const libc::c_char>::uninit();
        let failed = unsafe {
            ffi::LLVMLumenOptimize(
                module.as_ref(),
                target_machine.as_ref(),
                &self.config,
                error.as_mut_ptr(),
            )
        };
        if failed {
            let error = unsafe { CStr::from_ptr(error.assume_init()) };
            Err(anyhow!(error.to_str().unwrap()))
        } else {
            Ok(())
        }
    }
}

use std::mem::MaybeUninit;

use anyhow::anyhow;

use firefly_profiling::SelfProfilerRef;

use crate::profiling::{self, LlvmSelfProfiler};
use crate::support::OwnedStringRef;
use crate::target::TargetMachine;
use crate::{Module, OwnedModule};

use super::*;

/// Represents an instance of an LLVM pass manager
#[derive(Default)]
pub struct PassManager {
    config: OptimizerConfig,
    profiler: Option<SelfProfilerRef>,
}
impl PassManager {
    /// Create a new default pass manager
    #[inline]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable debug mode for this pass manager
    pub fn debug(&mut self, debug: bool) {
        self.config.debug = debug;
    }

    /// Enable pass verification
    pub fn verify(&mut self, verify: bool) {
        self.config.verify = verify;
    }

    /// Set the optimization level of this pass manager
    pub fn optimize(&mut self, level: PassBuilderOptLevel) {
        self.config.opt_level = level;
    }

    /// Set the optimizer stage this pass manager should run
    pub fn stage(&mut self, stage: OptStage) {
        self.config.opt_stage = stage;
    }

    /// Enable the MSan sanitizer
    pub fn sanitize_memory(&mut self, track_origins: bool) {
        self.config.sanitizer_opts.memory = true;
        self.config.sanitizer_opts.memory_track_origins = track_origins as u32;
    }

    /// Enable the TSan sanitizer
    pub fn sanitize_thread(&mut self) {
        self.config.sanitizer_opts.thread = true;
    }

    /// Enable the ASan sanitizer
    pub fn sanitize_address(&mut self) {
        self.config.sanitizer_opts.address = true;
    }

    /// Enabling profiling of this pass manager
    ///
    /// NOTE: If llvm_record_enabled is not true for the given profiler instance, this has no effect
    pub fn profile(&mut self, profiler: &SelfProfilerRef) {
        if profiler.llvm_recording_enabled() {
            self.profiler = Some(profiler.clone());
        }
    }

    /// Runs this pass manager on the given module, using the provided TargetMachine
    ///
    /// If successful, it returns the optimized/transformed module as an owned reference
    /// If not successful, it returns the error message produced by LLVM
    pub fn run(
        &self,
        module: OwnedModule,
        target_machine: TargetMachine,
    ) -> anyhow::Result<OwnedModule> {
        extern "C" {
            pub fn LLVMFireflyOptimize(
                module: Module,
                target_machine: TargetMachine,
                config: *const OptimizerConfig,
                error_out: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }

        let profiler = self.profiler.clone();
        let mut llvm_profiler = profiler
            .as_ref()
            .map(|p| LlvmSelfProfiler::new(p.get_self_profiler().unwrap()));
        let config = {
            let mut config = self.config.clone();
            let profiler = llvm_profiler
                .as_mut()
                .map(|p| p as *mut _ as *mut std::ffi::c_void)
                .unwrap_or(std::ptr::null_mut());
            config.profiler = profiler;
            config
        };

        let mut error = MaybeUninit::<*mut std::os::raw::c_char>::uninit();
        let failed = unsafe {
            LLVMFireflyOptimize(module.as_ref(), target_machine, &config, error.as_mut_ptr())
        };
        if failed {
            let error = unsafe { OwnedStringRef::from_ptr(error.assume_init()) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(module)
        }
    }
}

/// Represents the set of options passed to the optimizer during codegen
#[repr(C)]
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pipeline: *const std::os::raw::c_char,
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
    profiler: *mut std::ffi::c_void,
    before_pass: SelfProfileBeforePassCallback,
    after_pass: SelfProfileAfterPassCallback,
}
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            pipeline: std::ptr::null(),
            opt_level: PassBuilderOptLevel::default(),
            opt_stage: OptStage::default(),
            sanitizer_opts: SanitizerOptions::default(),
            debug: false,
            verify: false,
            use_thinlto_buffers: false,
            disable_simplify_lib_calls: false,
            emit_summary_index: false,
            emit_module_hash: false,
            preserve_use_list_order: false,
            profiler: std::ptr::null_mut(),
            before_pass: profiling::selfprofile_before_pass_callback,
            after_pass: profiling::selfprofile_after_pass_callback,
        }
    }
}

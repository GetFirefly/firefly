use liblumen_pass::Pass;
use liblumen_session::{Options, Sanitizer};

use crate::codegen;
use crate::target::TargetMachine;
use crate::OwnedModule;

use super::*;

/// Runs an LLVM pass manager pipeline as a Pass
pub struct PassManagerPass {
    manager: PassManager,
    target_machine: TargetMachine,
}
impl PassManagerPass {
    pub fn new(options: &Options, target_machine: TargetMachine) -> Self {
        let (speed, size) = codegen::to_llvm_opt_settings(options.opt_level);
        let opt_level = PassBuilderOptLevel::from_codegen_opts(speed, size);

        let mut manager = PassManager::new();
        manager.verify(options.debugging_opts.verify_llvm_ir);
        manager.debug(options.debug_assertions);
        manager.optimize(opt_level);

        for sanitizer in &options.debugging_opts.sanitizers {
            match sanitizer {
                Sanitizer::Memory => {
                    manager.sanitize_memory(options.debugging_opts.sanitizer_memory_track_origins)
                }
                Sanitizer::Thread => manager.sanitize_thread(),
                Sanitizer::Address => manager.sanitize_address(),
                _ => (),
            }
        }

        Self {
            manager,
            target_machine,
        }
    }
}
impl Pass for PassManagerPass {
    type Input<'a> = OwnedModule;
    type Output<'a> = OwnedModule;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        self.manager.run(module, self.target_machine)
    }
}

use anyhow::anyhow;
use liblumen_llvm as llvm;
use liblumen_pass::Pass;

use crate::{Module, OwnedModule, StringRef};

pub struct TranslateMLIRToLLVMIR {
    llvm_context: llvm::Context,
    source_name: String,
}
impl<'c> TranslateMLIRToLLVMIR {
    pub fn new(llvm_context: llvm::Context, source_name: String) -> Self {
        Self {
            llvm_context,
            source_name,
        }
    }
}

impl Pass for TranslateMLIRToLLVMIR {
    type Input<'a> = &'a OwnedModule;
    type Output<'a> = llvm::OwnedModule;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        let module_name = module
            .name()
            .ok_or_else(|| anyhow!("expected module to be named"))?;
        let source_name = StringRef::from(self.source_name.as_str());

        let llvm_module = unsafe {
            let llvm_module =
                mlir_translate_module_to_llvm_ir(**module, self.llvm_context, source_name);
            if llvm_module.is_null() {
                return Err(anyhow!("failed to lower mlir module to llvm ir"));
            }
            llvm_module
        };
        llvm_module.set_name(module_name);
        llvm_module.set_source_file(source_name);

        Ok(llvm_module)
    }
}

extern "C" {
    #[link_name = "mlirTranslateModuleToLLVMIR"]
    fn mlir_translate_module_to_llvm_ir(
        module: Module,
        llvm_context: llvm::Context,
        name: StringRef,
    ) -> llvm::OwnedModule;
}

mod llvm;

pub use self::llvm::TranslateMLIRToLLVMIR;

use crate::Context;

/// Registers all translation passes with the given MLIR context
pub fn register_all_translations(context: Context) {
    unsafe {
        mlir_register_all_llvm_translations(context);
    }
}

extern "C" {
    #[link_name = "mlirRegisterAllLLVMTranslations"]
    fn mlir_register_all_llvm_translations(context: Context);
}

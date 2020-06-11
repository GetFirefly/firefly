use crate::sys::prelude::*;

extern "C" {
    pub fn LLVMAddCoroEarlyPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCoroSplitPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCoroElidePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCoroCleanupPass(PM: LLVMPassManagerRef);
}

use crate::sys::prelude::*;

extern "C" {
    pub fn LLVMAddAggressiveInstCombinerPass(PM: LLVMPassManagerRef);
}

//! Interprocedural transformations of LLVM IR.

use super::super::prelude::*;

extern "C" {
    pub fn LLVMAddArgumentPromotionPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddConstantMergePass(PM: LLVMPassManagerRef);
    pub fn LLVMAddCalledValuePropagationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddDeadArgEliminationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddFunctionAttrsPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddFunctionInliningPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddAlwaysInlinerPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddGlobalDCEPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddGlobalOptimizerPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddIPConstantPropagationPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddPruneEHPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddIPSCCPPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddInternalizePass(arg1: LLVMPassManagerRef, AllButMain: ::libc::c_uint);
    pub fn LLVMAddStripDeadPrototypesPass(PM: LLVMPassManagerRef);
    pub fn LLVMAddStripSymbolsPass(PM: LLVMPassManagerRef);
}

use llvm_sys::{LLVMLinkage, LLVMThreadLocalMode};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Private,
    Internal,
    External,
    Weak,
}
impl Default for Linkage {
    fn default() -> Self {
        Self::External
    }
}
impl Into<LLVMLinkage> for Linkage {
    fn into(self) -> LLVMLinkage {
        match self {
            Self::Private => LLVMLinkage::LLVMPrivateLinkage,
            Self::Internal => LLVMLinkage::LLVMInternalLinkage,
            Self::External => LLVMLinkage::LLVMExternalLinkage,
            Self::Weak => LLVMLinkage::LLVMWeakAnyLinkage,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadLocalMode {
    NotThreadLocal,
    GeneralDynamic,
    LocalDynamic,
    InitialExec,
    LocalExec,
}
impl Default for ThreadLocalMode {
    fn default() -> Self {
        Self::NotThreadLocal
    }
}
impl Into<LLVMThreadLocalMode> for ThreadLocalMode {
    fn into(self) -> LLVMThreadLocalMode {
        match self {
            Self::NotThreadLocal => LLVMThreadLocalMode::LLVMNotThreadLocal,
            Self::GeneralDynamic => LLVMThreadLocalMode::LLVMGeneralDynamicTLSModel,
            Self::LocalDynamic => LLVMThreadLocalMode::LLVMLocalDynamicTLSModel,
            Self::InitialExec => LLVMThreadLocalMode::LLVMInitialExecTLSModel,
            Self::LocalExec => LLVMThreadLocalMode::LLVMLocalExecTLSModel,
        }
    }
}

use std::fmt;

use crate::sys::{LLVMLinkage, LLVMThreadLocalMode};

#[derive(Copy, Clone)]
#[repr(C)]
pub enum FileType {
    #[allow(dead_code)]
    Other,
    AssemblyFile,
    ObjectFile,
}

/// LLVMCodeGenOptLevel
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub enum CodeGenOptLevel {
    Other,
    None,
    Less,
    Default,
    Aggressive,
}
impl fmt::Display for CodeGenOptLevel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::None => f.write_str("none"),
            Self::Less => f.write_str("less"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd)]
#[repr(C)]
pub enum CodeGenOptSize {
    Other,
    None,
    Default,
    Aggressive,
}
impl fmt::Display for CodeGenOptSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Other => f.write_str("other"),
            Self::None => f.write_str("none"),
            Self::Default => f.write_str("default"),
            Self::Aggressive => f.write_str("aggressive"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    Private,
    Internal,
    External,
    Weak,
    LinkOnceODR,
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
            Self::LinkOnceODR => LLVMLinkage::LLVMLinkOnceODRLinkage,
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

pub fn to_llvm_opt_settings(cfg: liblumen_session::OptLevel) -> (CodeGenOptLevel, CodeGenOptSize) {
    use liblumen_session::OptLevel;
    match cfg {
        OptLevel::No => (CodeGenOptLevel::None, CodeGenOptSize::None),
        OptLevel::Less => (CodeGenOptLevel::Less, CodeGenOptSize::None),
        OptLevel::Default => (CodeGenOptLevel::Default, CodeGenOptSize::None),
        OptLevel::Aggressive => (CodeGenOptLevel::Aggressive, CodeGenOptSize::None),
        OptLevel::Size => (CodeGenOptLevel::Default, CodeGenOptSize::Default),
        OptLevel::SizeMin => (CodeGenOptLevel::Default, CodeGenOptSize::Aggressive),
    }
}

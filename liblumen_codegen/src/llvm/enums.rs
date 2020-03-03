use llvm_sys::LLVMLinkage;

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

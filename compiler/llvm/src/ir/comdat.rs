extern "C" {
    type LlvmComdat;
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ComdatSelectionKind {
    /// The linker may choose any COMDAT
    Any = 0,
    /// The data referenced by the COMDAT must be the same
    ExactMatch,
    /// The linker will choose the largest COMDAT
    Largest,
    /// No deduplication is performed
    NoDedup,
    /// The data referenced by the COMDAT must be the same size
    SameSize,
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Comdat(*const LlvmComdat);
impl Comdat {
    /// Get the conflict resolution selection kind for this COMDAT
    pub fn kind(self) -> ComdatSelectionKind {
        extern "C" {
            fn LLVMGetComdatSelectionKind(comdat: Comdat) -> ComdatSelectionKind;
        }
        unsafe { LLVMGetComdatSelectionKind(self) }
    }

    /// Set the conflict resolution selection kind for this COMDAT
    pub fn set_kind(self, kind: ComdatSelectionKind) {
        extern "C" {
            fn LLVMSetComdatSelectionKind(comdat: Comdat, kind: ComdatSelectionKind);
        }
        unsafe { LLVMSetComdatSelectionKind(self, kind) }
    }
}

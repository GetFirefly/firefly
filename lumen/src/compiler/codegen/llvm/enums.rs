use llvm_sys::target_machine::{LLVMCodeGenFileType, LLVMCodeGenOptLevel};

/// Represents the type of output to generate
pub enum OutputType {
    IR,
    Assembly,
    Object,
}
impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            OutputType::IR => f.write_str("ll"),
            OutputType::Assembly => f.write_str("s"),
            OutputType::Object => f.write_str("o"),
        }
    }
}
impl std::convert::From<LLVMCodeGenFileType> for OutputType {
    fn from(ty: LLVMCodeGenFileType) -> Self {
        match ty {
            LLVMCodeGenFileType::LLVMAssemblyFile => OutputType::Assembly,
            LLVMCodeGenFileType::LLVMObjectFile => OutputType::Object,
        }
    }
}
impl std::convert::Into<LLVMCodeGenFileType> for OutputType {
    fn into(self) -> LLVMCodeGenFileType {
        match self {
            OutputType::Assembly => LLVMCodeGenFileType::LLVMAssemblyFile,
            OutputType::Object => LLVMCodeGenFileType::LLVMObjectFile,
            OutputType::IR => panic!("LLVMCodeGenFileType does not support the IR type"),
        }
    }
}

/// Represents the amount of optimization to apply during codegen
pub enum Optimization {
    None,
    Less,
    Default,
    Aggressive,
}
impl std::convert::From<LLVMCodeGenOptLevel> for Optimization {
    fn from(level: LLVMCodeGenOptLevel) -> Self {
        match level {
            LLVMCodeGenOptLevel::LLVMCodeGenLevelNone => Optimization::None,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelLess => Optimization::Less,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault => Optimization::Default,
            LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive => Optimization::Aggressive,
        }
    }
}
impl std::convert::Into<LLVMCodeGenOptLevel> for Optimization {
    fn into(self) -> LLVMCodeGenOptLevel {
        match self {
            Optimization::None => LLVMCodeGenOptLevel::LLVMCodeGenLevelNone,
            Optimization::Less => LLVMCodeGenOptLevel::LLVMCodeGenLevelLess,
            Optimization::Default => LLVMCodeGenOptLevel::LLVMCodeGenLevelDefault,
            Optimization::Aggressive => LLVMCodeGenOptLevel::LLVMCodeGenLevelAggressive,
        }
    }
}

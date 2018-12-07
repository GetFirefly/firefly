#[derive(Debug)]
pub enum CodeGenError {
    ValidationError(String),
    LLVMError(String),
}
impl CodeGenError {
    pub fn new(reason: &str) -> CodeGenError {
        CodeGenError::LLVMError(reason)
    }
}
impl std::fmt::Display for CodeGenError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::CodeGenError::*;
        match *self {
            ValidationError(ref e) => write!(f, "invalid codegen input: {}", e),
            LLVMError(ref e) => write!(f, "LLVM failed: {}", e),
        }
    }
}
impl std::error::Error for CodeGenError {
    fn description(&self) -> &str {
        use self::CodeGenError::*;
        match *self {
            ValidationError(ref e) => e,
            LLVMError(ref e) => e,
        }
    }
}

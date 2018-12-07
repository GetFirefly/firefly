use std::path::Path;
use std::ffi::{CString,CStr};

use syntax::ast::ast::ModuleDecl;
use libc;

// Helper macro for converting String/str to a C string pointer
macro_rules! c_str {
    ($s:expr) => (
        CString::new($s).expect("invalid cast to C string").as_ptr() as *const i8
    );
}

// Helper macro for converting from a C string pointer to &str
macro_rules! c_str_to_str {
    ($s:expr) => (
        unsafe { CStr::from_ptr($s).to_str().expect("invalid C string pointer") }
    )
}

mod error;
mod module;
mod context;
mod generator;
mod target;
mod utils;

pub use self::error::CodeGenError;
pub use self::module::Module;

use self::generator::CodeGenerator;

/// The type of output to emit when generating code
pub enum OutputType {
    IR,
    Assembly,
    Object
}
impl OutputType {
    pub fn to_extension(&self) -> &str {
        match *self {
            OutputType::IR => "ll",
            OutputType::Assembly => "s",
            OutputType::Object => "o"
        }
    }
}
impl std::fmt::Display for OutputType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_extension())
    }
}

/// The level of optimization to apply to generated code
pub enum OptLevel {
    None,
    Less,
    Default,
    Aggressive
}

/// This trait is used to decorate AST nodes so that they can generate code for themselves
/// A parent node recursively invokes the generate callback of their children
pub trait CodeGen {
    fn generate(&self, &Module) -> Result<(), CodeGenError>;
}

/// Generate the IR for the provided modules to the given path in the given output type
pub fn generate_to_file(modules: &[ModuleDecl], out: Path, out_type: OutputType) -> Result<(), CodeGenError> {
    let codegen = CodeGenerator::new()?;
    for module in modules.iter() {
        codegen.module(module)?;
    }
    codegen.emit_to_file(out, out_type)
}

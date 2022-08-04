use std::path::PathBuf;
use std::sync::Arc;
use std::thread::ThreadId;

use liblumen_llvm as llvm;
use liblumen_mlir as mlir;
use liblumen_session::{InputType, Options};
use liblumen_syntax_core as syntax_core;
use liblumen_syntax_erl::{self as syntax_erl, ParseConfig};

use super::queries;
use crate::diagnostics::ErrorReported;
use crate::interner::*;
use crate::output::CompilerOutput;

#[salsa::query_group(ParserStorage)]
pub trait Parser: CompilerOutput {
    /// Returns the current compiler options
    #[salsa::input]
    fn options(&self) -> Arc<Options>;

    /// Returns configuration for the parser based on current compiler options
    #[salsa::invoke(queries::parse_config)]
    fn parse_config(&self) -> ParseConfig;

    /// Returns the output directory to which artifacts should be written
    #[salsa::invoke(queries::output_dir)]
    fn output_dir(&self) -> PathBuf;

    /// Returns the LLVM context associated with the given thread
    #[salsa::invoke(queries::llvm_context)]
    fn llvm_context(&self, thread_id: ThreadId) -> Arc<llvm::OwnedContext>;

    /// Returns the LLVM TargetMachine associated with the given thread
    #[salsa::invoke(queries::target_machine)]
    fn target_machine(&self, thread_id: ThreadId) -> Arc<llvm::target::OwnedTargetMachine>;

    /// Returns the MLIR context associated with the given thread
    #[salsa::invoke(queries::mlir_context)]
    fn mlir_context(&self, thread_id: ThreadId) -> Arc<mlir::OwnedContext>;

    /// Returns all of the input sources
    #[salsa::invoke(queries::inputs)]
    fn inputs(&self) -> Result<Vec<InternedInput>, ErrorReported>;

    /// Returns the type of an interned input
    #[salsa::invoke(queries::input_type)]
    fn input_type(&self, input: InternedInput) -> InputType;

    /// Gets the syntax_erl::ast module associated with the given input, if it exists
    ///
    /// If the input is not compatible with producing an AST module, or an
    /// error occurs during parsing of the module, the result will be Err(ErrorReported).
    #[salsa::invoke(queries::input_ast)]
    fn input_ast(&self, input: InternedInput) -> Result<syntax_erl::ast::Module, ErrorReported>;

    /// Gets the syntax_erl module associated with the given input, if it exists
    ///
    /// If the input is not compatible with producing a syntax_erl module, or an
    /// error occurs during parsing of the module, the result will be Err(ErrorReported).
    #[salsa::invoke(queries::input_cst)]
    fn input_cst(&self, input: InternedInput) -> Result<syntax_erl::cst::Module, ErrorReported>;

    #[salsa::invoke(queries::input_kernel)]
    fn input_kernel(&self, input: InternedInput) -> Result<syntax_erl::kst::Module, ErrorReported>;

    /// Gets the syntax_core module associated with the given input, if it exists
    ///
    /// If the input is not compatible with producing a syntax_core module, or an
    /// error occurs during parsing of the module, the result will be Err(ErrorReported).
    #[salsa::invoke(queries::input_syntax_core)]
    fn input_syntax_core(&self, input: InternedInput)
        -> Result<syntax_core::Module, ErrorReported>;

    /// Gets the mlir module associated with the given input, if it exists
    ///
    /// If the input is a .mlir file, this will parse directly to an mlir module, otherwise,
    /// higher-level sources (i.e. .erl/.core) will be parsed and then lowered to mlir.
    ///
    /// If an error occurred during parsing or lowering of the module, the result will be Err(ErrorReported).
    #[salsa::invoke(queries::input_mlir)]
    fn input_mlir(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> Result<mlir::OwnedModule, ErrorReported>;
}

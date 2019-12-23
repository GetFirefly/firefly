use std::sync::Arc;
use std::thread::ThreadId;

use liblumen_codegen::llvm;
use liblumen_codegen::mlir;
use liblumen_codegen::codegen::CompiledModule;
use liblumen_incremental::ParserDatabase;
use liblumen_incremental::{InternedInput, QueryResult};

use crate::compiler::intern::InternedString;
use crate::compiler::queries;

#[salsa::query_group(CodegenStorage)]
pub trait CodegenDatabase: CodegenDatabaseBase {
    #[salsa::invoke(queries::mlir_context)]
    fn mlir_context(&self, thread_id: ThreadId) -> Arc<mlir::Context>;

    #[salsa::invoke(queries::parse_mlir_module)]
    fn parse_mlir_module(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> QueryResult<Arc<mlir::Module>>;

    #[salsa::invoke(queries::generate_mlir)]
    fn generate_mlir(&self, thread_id: ThreadId, input: InternedInput) -> QueryResult<Arc<mlir::Module>>;

    #[salsa::invoke(queries::get_eir_dialect_module)]
    fn get_eir_dialect_module(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> QueryResult<Arc<mlir::Module>>;

    #[salsa::invoke(queries::get_llvm_dialect_module)]
    fn get_llvm_dialect_module(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> QueryResult<Arc<mlir::Module>>;

    #[salsa::invoke(queries::llvm_context)]
    fn llvm_context(&self, thread_id: ThreadId) -> Arc<llvm::Context>;

    #[salsa::invoke(queries::get_target_machine)]
    fn get_target_machine(&self, thread_id: ThreadId) -> Arc<llvm::TargetMachine>;

    #[salsa::invoke(queries::get_llvm_module)]
    fn get_llvm_module(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> QueryResult<Arc<llvm::Module>>;


    #[salsa::invoke(queries::compile)]
    fn compile(&self, input: InternedInput) -> QueryResult<Arc<CompiledModule>>;
}

#[salsa::query_group(StringInternerStorage)]
pub trait StringInternerDatabase: salsa::Database {
    #[salsa::interned]
    fn intern_string(&self, string: String) -> InternedString;
}

pub trait CodegenDatabaseBase: ParserDatabase + StringInternerDatabase {}

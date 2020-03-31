use std::collections::HashSet;
use std::sync::Arc;
use std::thread::ThreadId;

use liblumen_codegen::meta::CompiledModule;
use liblumen_core::symbols::FunctionSymbol;
use liblumen_incremental::ParserDatabase;
use liblumen_incremental::{InternedInput, QueryResult};
use liblumen_llvm as llvm;
use liblumen_mlir as mlir;

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
    fn generate_mlir(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
    ) -> QueryResult<Arc<mlir::Module>>;

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
    fn get_target_machine(&self, thread_id: ThreadId) -> Arc<llvm::target::TargetMachine>;

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

pub trait CodegenDatabaseBase: ParserDatabase + StringInternerDatabase {
    fn take_atoms(&mut self) -> HashSet<libeir_intern::Symbol>;
    fn add_atoms<'a, I>(&self, atoms: I)
    where
        I: Iterator<Item = &'a libeir_intern::Symbol>;
    fn take_symbols(&mut self) -> HashSet<FunctionSymbol>;
    fn add_symbols<'a, I>(&self, symbols: I)
    where
        I: Iterator<Item = &'a FunctionSymbol>;
}

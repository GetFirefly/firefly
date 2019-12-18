use liblumen_codegen::llvm;
use liblumen_codegen::mlir;
use liblumen_incremental::ParserDatabase;
use liblumen_incremental::{InternedInput, QueryResult};

use crate::compiler::intern::InternedString;
use crate::compiler::queries;

#[salsa::query_group(CodegenStorage)]
pub trait CodegenDatabase: CodegenDatabaseBase {
    #[salsa::invoke(queries::generate_mlir)]
    fn input_mlir(&self, input: InternedInput) -> QueryResult<mlir::Module<'static>>;

    #[salsa::invoke(queries::lower_eir_dialect)]
    fn input_standard_mlir(&self, input: InternedInput) -> QueryResult<mlir::Module<'static>>;

    #[salsa::invoke(queries::lower_standard_dialect)]
    fn input_llvm_mlir(&self, input: InternedInput) -> QueryResult<mlir::Module<'static>>;

    #[salsa::invoke(queries::generate_llvm_assembly)]
    fn input_llvm_ir(&self, input: InternedInput) -> QueryResult<llvm::Module<'static>>;

    #[salsa::invoke(queries::link)]
    fn link(&self) -> QueryResult<()>;
}

#[salsa::query_group(StringInternerStorage)]
pub trait StringInternerDatabase: salsa::Database {
    #[salsa::interned]
    fn intern_string(&self, string: String) -> InternedString;
}

pub trait CodegenDatabaseBase: ParserDatabase + StringInternerDatabase {}
impl<T> CodegenDatabaseBase for T where T: ParserDatabase + StringInternerDatabase {}

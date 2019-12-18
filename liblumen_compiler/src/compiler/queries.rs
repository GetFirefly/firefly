use liblumen_codegen::{self as codegen, llvm, mlir};
use liblumen_incremental::{InternedInput, QueryResult};

use crate::compiler::query_groups::CodegenDatabase;

pub(super) fn generate_mlir<C>(db: &C, input: InternedInput) -> QueryResult<mlir::Module<'static>>
where
    C: CodegenDatabase,
{
    let options = db.options();
    let module = db.input_eir(input)?;
    match codegen::generate_mlir(&options, &module) {
        Ok(mlir_module) => {
            db.maybe_emit_file_with_opts(&options, input, &mlir_module)?;
            Ok(mlir_module)
        }
        Err(err) => {
            db.diagnostics().error(err);
            Err(())
        }
    }
}

pub(super) fn lower_eir_dialect<C>(
    db: &C,
    input: InternedInput,
) -> QueryResult<mlir::Module<'static>>
where
    C: CodegenDatabase,
{
    let _module = db.input_mlir(input)?;
    unimplemented!();
}

pub(super) fn lower_standard_dialect<C>(
    db: &C,
    input: InternedInput,
) -> QueryResult<mlir::Module<'static>>
where
    C: CodegenDatabase,
{
    let _module = db.input_standard_mlir(input)?;
    unimplemented!();
}

pub(super) fn generate_llvm_assembly<C>(
    db: &C,
    input: InternedInput,
) -> QueryResult<llvm::Module<'static>>
where
    C: CodegenDatabase,
{
    let _module = db.input_llvm_mlir(input)?;
    unimplemented!();
}

pub(super) fn link<C>(_db: &C) -> QueryResult<()>
where
    C: CodegenDatabase,
{
    unimplemented!();
}

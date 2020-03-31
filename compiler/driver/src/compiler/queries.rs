use std::ops::Deref;
use std::sync::Arc;
use std::thread::{self, ThreadId};

use anyhow::anyhow;

use log::debug;

use liblumen_codegen::meta::CompiledModule;
use liblumen_codegen::{self as codegen, GeneratedModule};
use liblumen_incremental::{InternedInput, QueryResult};
use liblumen_llvm as llvm;
use liblumen_mlir as mlir;
use liblumen_session::{Input, InputType, OutputType};

use crate::compiler::query_groups::*;

macro_rules! to_query_result {
    ($db:expr, $val:expr) => {
        $val.map_err(|e| {
            $db.diagnostics().error(e);
            ()
        })?
    };
}

/// Create context for MLIR
pub(super) fn mlir_context<C>(db: &C, thread_id: ThreadId) -> Arc<mlir::Context>
where
    C: CodegenDatabase,
{
    use mlir::Context;

    debug!("constructing new mlir context for thread {:?}", thread_id);

    let options = db.options();
    let target_machine = db.get_target_machine(thread_id);
    Arc::new(Context::new(thread_id, &options, &target_machine))
}

/// Parse MLIR source file
pub(super) fn parse_mlir_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: CodegenDatabase,
{
    let input_info = db.lookup_intern_input(input);
    let context = db.mlir_context(thread_id);

    let parsed = match input_info {
        Input::File(ref path) => {
            debug!("parsing mlir from file for {:?} on {:?}", input, thread_id);
            to_query_result!(db, context.parse_file(path))
        }
        Input::Str {
            ref name,
            ref input,
            ..
        } => {
            debug!(
                "parsing mlir from string for {:?} on {:?}",
                input, thread_id
            );
            to_query_result!(db, context.parse_string(input, name))
        }
    };
    Ok(Arc::new(parsed))
}

/// Convert EIR to MLIR/EIR
pub(super) fn generate_mlir<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: CodegenDatabase,
{
    let module = db.input_eir(input)?;
    let context = db.mlir_context(thread_id);
    let options = db.options();
    debug!("generating mlir for {:?} on {:?}", input, thread_id);
    let target_machine = db.get_target_machine(thread_id);
    let filemap = {
        let codemap = db.codemap().read().unwrap();
        codemap
            .find_file(module.span().start())
            .map(|fm| fm.clone())
            .expect("expected input to have corresponding entry in code map")
    };
    match codegen::builder::build(&module, filemap, &context, &options, target_machine.deref()) {
        Ok(GeneratedModule {
            module: mlir_module,
            atoms,
            symbols,
        }) => {
            db.add_atoms(atoms.iter());
            db.add_symbols(symbols.iter());
            db.maybe_emit_file_with_opts(&options, input, &mlir_module)?;
            Ok(Arc::new(mlir_module))
        }
        Err(err) => {
            db.diagnostics().error(err);
            Err(())
        }
    }
}

/// Either load MLIR input directly, or lower EIR to MLIR, depending on type of input
pub(super) fn get_eir_dialect_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: CodegenDatabase,
{
    match db.input_type(input) {
        InputType::Erlang | InputType::AbstractErlang | InputType::EIR => {
            debug!("input {:?} is erlang", input);
            Ok(db.generate_mlir(thread_id, input)?)
        }
        InputType::MLIR => {
            debug!("input {:?} is mlir", input);
            Ok(db.parse_mlir_module(thread_id, input)?)
        }
        InputType::Unknown(None) => {
            debug!("unknown input type for {:?} on {:?}", input, thread_id);
            db.diagnostics()
                .error(anyhow!("invalid input, expected .erl or .mlir"));
            Err(())
        }
        InputType::Unknown(Some(ref ext)) => {
            debug!(
                "unsupported input type '{}' for {:?} on {:?}",
                ext, input, thread_id
            );
            db.diagnostics().error(anyhow!(
                "invalid input extension ({}), expected .erl or .mlir",
                ext
            ));
            Err(())
        }
    }
}

pub(super) fn get_llvm_dialect_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: CodegenDatabase,
{
    let options = db.options();
    let context = db.mlir_context(thread_id);
    let module = db.get_eir_dialect_module(thread_id, input)?;

    // Lower to LLVM dialect
    debug!(
        "lowering mlir to llvm dialect for {:?} on {:?}",
        input, thread_id
    );
    to_query_result!(db, module.lower(&context));

    // Emit LLVM dialect
    db.maybe_emit_file_with_opts(&options, input, module.deref())?;

    Ok(module)
}

/// Create context for LLVM
pub(super) fn llvm_context<C>(_db: &C, thread_id: ThreadId) -> Arc<llvm::Context>
where
    C: CodegenDatabase,
{
    use llvm::Context;
    debug!("constructing new llvm context for thread {:?}", thread_id);
    Arc::new(Context::new())
}

pub(super) fn get_target_machine<C>(db: &C, thread_id: ThreadId) -> Arc<llvm::target::TargetMachine>
where
    C: CodegenDatabase,
{
    let options = db.options();
    let diagnostics = db.diagnostics();
    debug!("constructing new target machine for thread {:?}", thread_id);
    Arc::new(llvm::target::create_target_machine(
        &options,
        diagnostics,
        false,
    ))
}

pub(super) fn get_llvm_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<llvm::Module>>
where
    C: CodegenDatabase,
{
    let options = db.options();
    let context = db.mlir_context(thread_id);
    let mlir_module = db.get_llvm_dialect_module(thread_id, input)?;

    // Convert to LLVM IR
    debug!("generating llvm for {:?} on {:?}", input, thread_id,);
    let source_name = get_input_source_name(db, input);
    let module = to_query_result!(db, mlir_module.lower_to_llvm_ir(&context, source_name));

    // Emit LLVM IR
    db.maybe_emit_file_with_opts(&options, input, &module)?;

    // Emit LLVM bitcode
    db.maybe_emit_file_with_callback_and_opts(
        &options,
        input,
        OutputType::LLVMBitcode,
        |outfile| {
            debug!("emitting llvm bitcode for {:?}", input);
            module.emit_bc(outfile)
        },
    )?;

    Ok(Arc::new(module))
}

pub(super) fn compile<C>(db: &C, input: InternedInput) -> QueryResult<Arc<CompiledModule>>
where
    C: CodegenDatabase,
{
    let thread_id = thread::current().id();

    let options = db.options();
    let input_info = db.lookup_intern_input(input);
    let diagnostics = db.diagnostics();

    diagnostics.success("Compiling", input_info.source_name());
    debug!(
        "compiling {:?} ({:?}) on thread {:?}",
        input, &input_info, thread_id
    );

    // Get LLVM IR module
    // We provide the current thread ID as part of the query, since the context
    // object of an LLVM module is not thread-safe, we only want to fulfill a
    // request for a module if the query occurs on the same thread
    let module = db.get_llvm_module(thread_id, input)?;

    // Emit textual assembly file
    db.maybe_emit_file_with_callback_and_opts(&options, input, OutputType::Assembly, |outfile| {
        debug!("emitting asm for {:?}", input);
        module.emit_asm(outfile)
    })?;

    // Emit object file
    let obj_path = db.maybe_emit_file_with_callback_and_opts(
        &options,
        input,
        OutputType::Object,
        |outfile| {
            debug!("emitting object file for {:?}", input);
            module.emit_obj(outfile)
        },
    )?;

    // Gather compiled module metadata
    let bc_path = options
        .output_types
        .maybe_emit(&input_info, OutputType::LLVMBitcode)
        .map(|filename| db.output_dir().join(filename));

    let compiled = Arc::new(CompiledModule::new(
        input_info.file_stem().to_string_lossy().into_owned(),
        obj_path,
        bc_path,
    ));

    debug!("compilation finished for {:?}", input);
    Ok(compiled)
}

fn get_input_source_name<C>(db: &C, input: InternedInput) -> Option<String>
where
    C: CodegenDatabase,
{
    let input_info = db.lookup_intern_input(input);

    match input_info {
        Input::File(ref path) => Some(path.to_string_lossy().into_owned()),
        Input::Str { ref name, .. } => Some(name.clone()),
    }
}

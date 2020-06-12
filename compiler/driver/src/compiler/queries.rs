use std::ops::Deref;
use std::sync::Arc;
use std::thread::{self, ThreadId};

use anyhow::anyhow;

use log::debug;

use liblumen_codegen::meta::CompiledModule;
use liblumen_codegen::{self as codegen, GeneratedModule};
use liblumen_llvm::{self as llvm, target::TargetMachineConfig};
use liblumen_mlir as mlir;
use liblumen_session::{Input, InputType, OutputType};

use super::prelude::*;

/// Create context for LLVM
pub(super) fn llvm_context<C>(db: &C, thread_id: ThreadId) -> Arc<llvm::Context>
where
    C: Compiler,
{
    use llvm::Context;
    debug!("constructing new llvm context for thread {:?}", thread_id);
    Arc::new(Context::new(db.diagnostics().clone()))
}

/// Create context for MLIR
pub(super) fn mlir_context<C>(db: &C, thread_id: ThreadId) -> Arc<mlir::Context>
where
    C: Compiler,
{
    use mlir::Context;

    debug!("constructing new mlir context for thread {:?}", thread_id);

    let options = db.options();
    let target_machine = db.get_target_machine(thread_id);
    let llvm_context = db.llvm_context(thread_id);

    Arc::new(Context::new(
        thread_id,
        &options,
        &llvm_context,
        &target_machine,
    ))
}

/// Parse MLIR source file
pub(super) fn parse_mlir_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: Compiler,
{
    let input_info = db.lookup_intern_input(input);
    let context = db.mlir_context(thread_id);

    let parsed = match input_info {
        Input::File(ref path) => {
            debug!("parsing mlir from file for {:?} on {:?}", input, thread_id);
            context.parse_file(path)
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
            context.parse_string(input, name)
        }
    };
    let parsed = db.to_query_result(parsed)?;
    Ok(Arc::new(parsed))
}

/// Convert EIR to MLIR/EIR
pub(super) fn generate_mlir<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: Compiler,
{
    use codegen::builder::build;

    let module = db.input_eir(input)?;
    let context = db.mlir_context(thread_id);
    let options = db.options();
    debug!("generating mlir for {:?} on {:?}", input, thread_id);
    let target_machine = db.get_target_machine(thread_id);
    let source_file = db
        .codemap()
        .get(module.span().start().source_id())
        .map(|s| s.clone())
        .expect("expected input to have corresponding entry in code map");
    let built = db.to_query_result(build(
        &module,
        source_file,
        &context,
        &options,
        target_machine.deref(),
    ))?;
    db.add_atoms(built.atoms.iter());
    db.add_symbols(built.symbols.iter());
    db.maybe_emit_file_with_opts(&options, input, &built.module)?;
    Ok(Arc::new(built.module))
}

/// Either load MLIR input directly, or lower EIR to MLIR, depending on type of input
pub(super) fn get_eir_dialect_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: Compiler,
{
    match db.input_type(input) {
        InputType::Erlang | InputType::AbstractErlang | InputType::EIR => {
            debug!("input {:?} is erlang", input);
            db.generate_mlir(thread_id, input)
        }
        InputType::MLIR => {
            debug!("input {:?} is mlir", input);
            db.parse_mlir_module(thread_id, input)
        }
        InputType::Unknown(None) => {
            debug!("unknown input type for {:?} on {:?}", input, thread_id);
            db.report_error("invalid input, expected .erl or .mlir");
            Err(ErrorReported)
        }
        InputType::Unknown(Some(ref ext)) => {
            debug!(
                "unsupported input type '{}' for {:?} on {:?}",
                ext, input, thread_id
            );
            db.report_error(format!(
                "invalid input extension ({}), expected .erl or .mlir",
                ext
            ));
            Err(ErrorReported)
        }
    }
}

pub(super) fn get_llvm_dialect_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<mlir::Module>>
where
    C: Compiler,
{
    let options = db.options();
    let context = db.mlir_context(thread_id);
    let module = db.get_eir_dialect_module(thread_id, input)?;

    // Lower to LLVM dialect
    debug!(
        "lowering mlir to llvm dialect for {:?} on {:?}",
        input, thread_id
    );
    db.to_query_result(module.lower(&context))?;

    // Emit LLVM dialect
    db.maybe_emit_file_with_opts(&options, input, module.deref())?;

    Ok(module)
}

pub(super) fn get_target_machine_config<C>(db: &C, thread_id: ThreadId) -> Arc<TargetMachineConfig>
where
    C: Compiler,
{
    let options = db.options();
    debug!(
        "constructing new target machine config for thread {:?}",
        thread_id
    );
    Arc::new(TargetMachineConfig::new(&options))
}

pub(super) fn get_target_machine<C>(db: &C, thread_id: ThreadId) -> Arc<llvm::target::TargetMachine>
where
    C: Compiler,
{
    let options = db.options();
    let diagnostics = db.diagnostics();
    let config = db.get_target_machine_config(thread_id);
    debug!("constructing new target machine for thread {:?}", thread_id);
    Arc::new(config.create().expect("failed to create target machine"))
}

pub(super) fn get_llvm_module<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> QueryResult<Arc<llvm::Module>>
where
    C: Compiler,
{
    use liblumen_llvm::passes::{PassBuilderOptLevel, PassManager};
    use liblumen_session::Sanitizer;

    let options = db.options();
    let context = db.mlir_context(thread_id);
    let mlir_module = db.get_llvm_dialect_module(thread_id, input)?;

    // Convert to LLVM IR
    debug!("generating llvm for {:?} on {:?}", input, thread_id,);
    let source_name = get_input_source_name(db, input);
    let mut module = db.to_query_result(mlir_module.lower_to_llvm_ir(&context, source_name))?;

    // Run optimizations
    let mut pass_manager = PassManager::new();
    pass_manager.verify(options.debugging_opts.verify_llvm_ir);
    pass_manager.debug(options.debug_assertions);
    let (speed, size) = llvm::enums::to_llvm_opt_settings(options.opt_level);
    pass_manager.optimize(PassBuilderOptLevel::from_codegen_opts(speed, size));
    if let Some(sanitizer) = options.debugging_opts.sanitizer {
        match sanitizer {
            Sanitizer::Memory => pass_manager.sanitize_memory(/* track_origins */ 0),
            Sanitizer::Thread => pass_manager.sanitize_thread(),
            Sanitizer::Address => pass_manager.sanitize_address(),
            _ => (),
        }
    }
    let target_machine = db.get_target_machine(thread_id);
    db.to_query_result(pass_manager.run(&mut module, &target_machine))?;

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
    C: Compiler,
{
    let thread_id = thread::current().id();

    let options = db.options();
    let input_info = db.lookup_intern_input(input);
    let source_name = input_info.source_name();
    let diagnostics = db.diagnostics();

    diagnostics.success("Compiling", format!("{}", &source_name));
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
    diagnostics.success("Compiled", format!("{}", &source_name));
    Ok(compiled)
}

fn get_input_source_name<C>(db: &C, input: InternedInput) -> Option<String>
where
    C: Compiler,
{
    let input_info = db.lookup_intern_input(input);

    match input_info {
        Input::File(ref path) => Some(path.to_string_lossy().into_owned()),
        Input::Str { ref name, .. } => Some(name.clone()),
    }
}

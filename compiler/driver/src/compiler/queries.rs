use std::thread::ThreadId;

use log::debug;

use liblumen_codegen::meta::CompiledModule;
use liblumen_intern::Symbol;
use liblumen_session::OutputType;

use super::prelude::*;

macro_rules! unwrap_or_bail {
    ($db:ident, $e:expr) => {
        match $e {
            Ok(result) => result,
            Err(ref e) => {
                bail!($db, "{}", e);
            }
        }
    };
}

macro_rules! bail {
    ($db:ident, $e:expr) => {{
        $db.report_error($e);
        return Err(ErrorReported);
    }};

    ($db:ident, $fmt:literal, $($arg:expr),*) => {{
        $db.report_error(format!($fmt, $($arg),*));
        return Err(ErrorReported);
    }}
}

pub(super) fn compile<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
) -> Result<Option<CompiledModule>, ErrorReported>
where
    C: Compiler,
{
    use liblumen_llvm::passes::PassManagerPass;
    use liblumen_mlir::translations::TranslateMLIRToLLVMIR;
    use liblumen_mlir::{PassManager, PassManagerOptions};
    use liblumen_pass::Pass;

    let options = db.options();
    let input_info = db.lookup_intern_input(input);
    let source_name = input_info.source_name();
    let diagnostics = db.diagnostics();
    let mlir_context = db.mlir_context(thread_id);
    let llvm_context = db.llvm_context(thread_id);
    let target_machine = db.target_machine(thread_id);
    let data_layout = target_machine.data_layout();

    // Fetch the application to be compiled
    diagnostics.success("Compiling", format!("{}", &source_name));
    debug!("compiling {} on thread {:?}", &source_name, thread_id);

    // Bail early if we don't have artifacts to codegen
    if !options.output_types.should_generate_mlir() {
        // However, since production of AST/CoreIR is driven by
        // queries for MLIR, we need to check if either of those
        // types were requested, and if so, execute the appropriate
        // query
        if options.output_types.should_generate_core() {
            db.input_syntax_core(input)?;
        } else if options.output_types.contains_key(&OutputType::AST) {
            db.input_syntax_erl(input)?;
        }
        return Ok(None);
    }

    let module = db.input_mlir(thread_id, input)?;
    let data_layout_str = data_layout.to_string();
    module.set_data_layout(data_layout_str.as_str());
    module.set_target_triple(target_machine.triple());
    let module_name = module
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|| input_info.file_stem());
    let module_sym = Symbol::intern(module_name.as_str());

    // Create a pass manager to perform the lowering
    let pm_opts = PassManagerOptions::new(&options);
    let mut pm = PassManager::new(**mlir_context, &pm_opts);
    //let mpm = pm.nest("builtin.module");
    //mpm.add(liblumen_mlir::conversions::ConvertCIRToLLVMPass::new());
    liblumen_mlir::conversions::ConvertCIRToLLVMPass::register();
    pm.parse_pipeline("builtin.module(convert-cir-to-llvm)")
        .unwrap();

    // Lower to LLVM dialect
    let successful = pm.run(&module);
    if !successful {
        db.report_error(format!(
            "error occurred while lowering this module to llvm dialect {}",
            &module_name
        ));
        return Err(ErrorReported);
    }
    db.maybe_emit_file_with_opts(&options, input, &module)?;

    // Convert to LLVM IR, or bail early if MLIR is all that was requested
    if !options.output_types.should_generate_llvm() {
        return Ok(None);
    }
    debug!("generating llvm for {:?} on {:?}", input, thread_id);
    let mut translation = TranslateMLIRToLLVMIR::new(llvm_context.borrow(), module_name.clone());
    let module = unwrap_or_bail!(db, translation.run(&module));

    // Verify/optimize
    let mut optimizer = PassManagerPass::new(&options, target_machine.handle());
    let module = unwrap_or_bail!(db, optimizer.run(module));

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

    // Emit textual assembly file
    db.maybe_emit_file_with_callback_and_opts(&options, input, OutputType::Assembly, |outfile| {
        debug!("emitting asm for {:?}", input);
        module.emit_asm(outfile, target_machine.handle())
    })?;

    // Emit object file
    let obj_path = db.maybe_emit_file_with_callback_and_opts(
        &options,
        input,
        OutputType::Object,
        |outfile| {
            debug!("emitting object file for {:?}", input);
            module.emit_obj(outfile, target_machine.handle())
        },
    )?;

    // Gather compiled module metadata
    let bc_path = options
        .output_types
        .maybe_emit(&input_info, OutputType::LLVMBitcode)
        .map(|filename| db.output_dir().join(filename));

    let compiled = CompiledModule::new(module_sym, obj_path, bc_path);

    debug!("compilation finished for {:?}", input);
    diagnostics.success("Compiled", format!("{}", &module_name));
    Ok(Some(compiled))
}

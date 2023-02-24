#[cfg(feature = "native-compilation")]
pub(super) fn compile<C>(
    db: &C,
    thread_id: ThreadId,
    input: InternedInput,
    app: Arc<ApplicationMetadata>,
) -> Result<Option<CompiledModule>, ErrorReported>
where
    C: Compiler,
{
    use firefly_intern::Symbol;
    use firefly_llvm::passes::PassManagerPass;
    use firefly_mlir::translations::TranslateMLIRToLLVMIR;
    use firefly_mlir::{PassManager, PassManagerOptions};
    use firefly_pass::Pass;

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

    // Bail early if we are just performing analysis or don't have artifacts to codegen
    if options.debugging_opts.analyze_only {
        // We conduct analysis all the way up through Kernel, so run at least that much
        db.input_kernel(input, app)?;
        return Ok(None);
    } else if !options.output_types.should_generate_mlir() {
        // Since production of Core/Kernel/SSA IR is driven by
        // queries for MLIR, we need to check if any of those
        // types were requested, and if so, execute the appropriate query
        if options.output_types.should_generate_ssa() {
            db.input_ssa(input, app)?;
        } else if options.output_types.contains_key(&OutputType::Kernel) {
            db.input_kernel(input, app)?;
        } else if options.output_types.contains_key(&OutputType::Core) {
            db.input_core(input, app)?;
        }
        return Ok(None);
    }

    let module = db.input_mlir(thread_id, input, app)?;

    // Bail prior to lowering CIR dialect to LLVM dialect if we aren't
    // going to generate LLVM IR
    if !options.output_types.should_generate_llvm() {
        return Ok(None);
    }

    debug!(
        "converting cir dialect to llvm dialect for {:?} on {:?}",
        input, thread_id
    );
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
    //mpm.add(firefly_mlir::conversions::ConvertCIRToLLVMPass::new());
    firefly_mlir::conversions::ConvertCIRToLLVMPass::register();
    pm.parse_pipeline("convert-cir-to-llvm,reconcile-unrealized-casts")
        .unwrap();

    // Lower to LLVM dialect
    let successful = pm.run(&module);
    if !successful {
        use firefly_mlir::Operation;
        module.as_ref().dump();

        diagnostics.error(format!(
            "error occurred while lowering module '{}' to llvm dialect",
            &module_name
        ));
        return Err(ErrorReported);
    }
    db.maybe_emit_file_with_opts(&options, input, &module)?;

    debug!("generating llvm for {:?} on {:?}", input, thread_id);
    let mut translation =
        TranslateMLIRToLLVMIR::new(llvm_context.borrow(), source_name.to_string());
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

    // Bail early if we don't plan to run the code generator
    if !options.output_types.should_codegen() {
        return Ok(None);
    }

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

    let compiled = CompiledModule {
        name: module_sym,
        object: obj_path,
        dwarf_object: None,
        bytecode: bc_path,
    };

    debug!("compilation finished for {:?}", input);
    diagnostics.success("Compiled", format!("{}", &module_name));
    Ok(Some(compiled))
}

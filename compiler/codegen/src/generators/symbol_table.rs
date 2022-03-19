use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_intern::Symbol;
use liblumen_llvm::builder::ModuleBuilder;
use liblumen_llvm::ir::*;
use liblumen_llvm::target::TargetMachine;
use liblumen_session::{Input, Options, OutputType};
use liblumen_syntax_core::FunctionName;

use crate::meta::CompiledModule;

/// Generates an LLVM module containing the raw symbol table data for the current build
///
/// This is similar to the atom table generation, but simpler, in that we just generate
/// a large list of `FunctionSybmol` structs, which reference extern declarations of all
/// the functions defined by the build. At link time these will be resolved to pointers
/// to the actual functions, and when we boot the runtime, we can reify this array into
/// a more efficient search structure for dispatch.
pub fn generate(
    options: &Options,
    context: &Context,
    target_machine: TargetMachine,
    symbols: HashSet<FunctionSymbol>,
) -> anyhow::Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_dispatch";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    fn declare_extern_symbol<'ctx>(
        builder: &ModuleBuilder<'ctx>,
        symbol: &FunctionSymbol,
    ) -> anyhow::Result<Function> {
        let ms = Symbol::new(symbol.module as u32);
        let fs = Symbol::new(symbol.function as u32);
        let ident = FunctionName::new(ms, fs, symbol.arity);
        let name = ident.to_string();
        let ty = builder.get_erlang_function_type(ident.arity);
        Ok(builder.build_external_function(name.as_str(), ty))
    }

    // Translate FunctionName to FunctionSymbol with pointer to declared function
    let usize_type = builder.get_usize_type();
    let i8_type = builder.get_i8_type();
    let fn_ptr_type = builder.get_pointer_type(builder.get_opaque_function_type());
    let function_symbol_type = builder.get_struct_type(
        Some("FunctionSymbol"),
        &[
            usize_type.base(),
            usize_type.base(),
            i8_type.base(),
            fn_ptr_type.base(),
        ],
    );

    // Build values for array
    let mut functions: Vec<ConstantValue> = Vec::with_capacity(symbols.len());
    for symbol in symbols.iter() {
        let decl = declare_extern_symbol(&builder, symbol)?;
        let decl_ptr = ConstantExpr::pointer_cast(decl, fn_ptr_type);
        let module = builder.build_constant_uint(usize_type, symbol.module as u64);
        let fun = builder.build_constant_uint(usize_type, symbol.function as u64);
        let arity = builder.build_constant_uint(i8_type, symbol.arity as u64);
        let function = builder.build_constant_named_struct(
            function_symbol_type,
            &[module.into(), fun.into(), arity.into(), decl_ptr.into()],
        );
        functions.push(function.into());
    }

    // Generate global array of all idents
    let functions_const_init =
        builder.build_constant_array(function_symbol_type, functions.as_slice());
    let functions_const_ty = functions_const_init.get_type();
    let functions_const = builder.build_constant(
        functions_const_ty,
        "__LUMEN_SYMBOL_TABLE_ENTRIES",
        functions_const_init,
    );
    functions_const.set_linkage(Linkage::Private);
    functions_const.set_alignment(8);
    let functions_const: ConstantValue = functions_const.try_into().unwrap();

    let function_ptr_type = builder.get_pointer_type(function_symbol_type);
    let table_global_init =
        builder.build_const_inbounds_gep(functions_const_ty, functions_const, &[0, 0]);
    let table_global = builder.build_global(
        function_ptr_type,
        "__LUMEN_SYMBOL_TABLE",
        Some(table_global_init.base()),
    );
    table_global.set_alignment(8);

    // Generate array length global
    let table_size_global_init = builder.build_constant_uint(usize_type, functions.len() as u64);
    let table_size_global = builder.build_global(
        usize_type,
        "__LUMEN_SYMBOL_TABLE_SIZE",
        Some(table_size_global_init.base()),
    );
    table_size_global.set_alignment(8);

    // Generate thread local variable for current reduction count
    let i32_type = builder.get_i32_type();
    let reduction_count_init = builder.build_constant_uint(i32_type, 0);
    let reduction_count_global = builder.build_global(
        i32_type,
        "CURRENT_REDUCTION_COUNT",
        Some(reduction_count_init.base()),
    );
    reduction_count_global.set_thread_local(true);
    reduction_count_global.set_thread_local_mode(ThreadLocalMode::LocalExec);
    reduction_count_global.set_linkage(Linkage::External);
    reduction_count_global.set_alignment(8);

    // Generate thread local variable for process signal
    let process_signal_init = builder.build_constant_uint(i8_type, 0);
    let process_signal_global = builder.build_global(
        i8_type,
        "__lumen_process_signal",
        Some(process_signal_init.base()),
    );
    process_signal_global.set_thread_local(true);
    process_signal_global.set_thread_local_mode(ThreadLocalMode::LocalExec);
    process_signal_global.set_linkage(Linkage::External);
    process_signal_global.set_alignment(8);

    // We have to build a shim for the Rust libstd `lang_start_internal`
    // function to start the Rust runtime. Since that symbol is internal,
    // we locate the mangled symbol name at build time and build a shim
    // function that calls it while exporting itself with a non-mangled name
    //
    // We do that here, since logically its another symbol in our symbol table,
    // except we call it directly like any other function in the generated code.
    let lang_start_symbol_name = env!("LANG_START_SYMBOL_NAME");
    let lang_start_alias_name = "__lumen_lang_start_internal";

    let i32_type = builder.get_i32_type();
    let i8ptrptr_type = builder.get_pointer_type(builder.get_pointer_type(i8_type));
    let main_ptr_ty = builder.get_pointer_type(builder.get_function_type(i32_type, &[], false));
    let lang_start_ty = builder.get_function_type(
        usize_type,
        &[main_ptr_ty.base(), usize_type.base(), i8ptrptr_type.base()],
        /* variadic= */ false,
    );

    let lang_start_fn_decl = builder.build_external_function(lang_start_symbol_name, lang_start_ty);
    let lang_start_shim_fn = builder.build_external_function(lang_start_alias_name, lang_start_ty);

    let entry_block = builder.build_entry_block(lang_start_shim_fn);
    builder.position_at_end(entry_block);

    let lang_start_args = lang_start_shim_fn.arguments();
    let forwarded_args = lang_start_args.iter().map(|v| v.base()).collect::<Vec<_>>();
    let lang_start_call = builder.build_call(lang_start_fn_decl, forwarded_args.as_slice(), None);
    lang_start_call.set_tail_call(true);
    builder.build_return(lang_start_call);

    // Finalize module
    let module = builder.finish()?;

    // We need an input to represent the generated source
    let input = Input::from(Path::new(&format!("{}", NAME)));

    // Emit LLVM IR file
    if let Some(ir_path) = options.maybe_emit(&input, OutputType::LLVMAssembly) {
        let mut file = File::create(ir_path.as_path())?;
        module.emit_ir(&mut file)?;
    }

    // Emit LLVM bitcode file
    if let Some(bc_path) = options.maybe_emit(&input, OutputType::LLVMBitcode) {
        let mut file = File::create(bc_path.as_path())?;
        module.emit_bc(&mut file)?;
    }

    // Emit assembly file
    if let Some(asm_path) = options.maybe_emit(&input, OutputType::Assembly) {
        let mut file = File::create(asm_path.as_path())?;
        module.emit_asm(&mut file, target_machine)?;
    }

    // Emit object file
    let obj_path = if let Some(obj_path) = options.maybe_emit(&input, OutputType::Object) {
        let mut file = File::create(obj_path.as_path())?;
        module.emit_obj(&mut file, target_machine)?;
        Some(obj_path)
    } else {
        None
    };

    Ok(Arc::new(CompiledModule::new(
        NAME.to_string(),
        obj_path,
        None,
    )))
}

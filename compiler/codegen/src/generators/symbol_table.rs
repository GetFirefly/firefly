use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::mem;
use std::path::Path;
use std::sync::Arc;

use libeir_intern::{Ident, Symbol};
use libeir_ir::FunctionIdent;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_llvm as llvm;
use liblumen_llvm::builder::ModuleBuilder;
use liblumen_llvm::enums::{Linkage, ThreadLocalMode};
use liblumen_llvm::target::TargetMachine;
use liblumen_session::Options;

use crate::meta::CompiledModule;
use crate::Result;

/// Generates an LLVM module containing the raw symbol table data for the current build
///
/// This is similar to the atom table generation, but simpler, in that we just generate
/// a large list of `FunctionSybmol` structs, which reference extern declarations of all
/// the functions defined by the build. At link time these will be resolved to pointers
/// to the actual functions, and when we boot the runtime, we can reify this array into
/// a more efficient search structure for dispatch.
pub fn generate(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
    symbols: HashSet<FunctionSymbol>,
    output_dir: &Path,
) -> Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_dispatch";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    fn declare_extern_symbol<'ctx>(
        builder: &ModuleBuilder<'ctx>,
        symbol: &FunctionSymbol,
    ) -> Result<llvm::Value> {
        let ms = unsafe { mem::transmute::<u32, Symbol>(symbol.module as u32) };
        let fs = unsafe { mem::transmute::<u32, Symbol>(symbol.function as u32) };
        let ident = FunctionIdent {
            module: Ident::with_empty_span(ms),
            name: Ident::with_empty_span(fs),
            arity: symbol.arity as usize,
        };
        let name = ident.to_string();
        let ty = builder.get_erlang_function_type(ident.arity);
        Ok(builder.build_external_function(name.as_str(), ty))
    }

    // Translate FunctionIdent to FunctionSymbol with pointer to declared function
    let usize_type = builder.get_usize_type();
    let i8_type = builder.get_i8_type();
    let fn_ptr_type = builder.get_pointer_type(builder.get_opaque_function_type());
    let function_symbol_type = builder.get_struct_type(
        Some("FunctionSymbol"),
        &[usize_type, usize_type, i8_type, fn_ptr_type],
    );

    // Build values for array
    let mut functions = Vec::with_capacity(symbols.len());
    for symbol in symbols.iter() {
        let decl = declare_extern_symbol(&builder, symbol)?;
        let decl_ptr = builder.build_pointer_cast(decl, fn_ptr_type);
        let module = builder.build_constant_uint(usize_type, symbol.module as u64);
        let fun = builder.build_constant_uint(usize_type, symbol.function as u64);
        let arity = builder.build_constant_uint(i8_type, symbol.arity as u64);
        let function =
            builder.build_constant_struct(function_symbol_type, &[module, fun, arity, decl_ptr]);
        functions.push(function);
    }

    // Generate global array of all idents
    let functions_const_init =
        builder.build_constant_array(function_symbol_type, functions.as_slice());
    let functions_const_ty = builder.type_of(functions_const_init);
    let functions_const = builder.build_constant(
        functions_const_ty,
        "__LUMEN_SYMBOL_TABLE_ENTRIES",
        Some(functions_const_init),
    );
    builder.set_linkage(functions_const, Linkage::Private);
    builder.set_alignment(functions_const, 8);

    let function_ptr_type = builder.get_pointer_type(function_symbol_type);
    let table_global_init = builder.build_const_inbounds_gep(functions_const, &[0, 0]);
    let table_global = builder.build_global(
        function_ptr_type,
        "__LUMEN_SYMBOL_TABLE",
        Some(table_global_init),
    );
    builder.set_alignment(table_global, 8);

    // Generate array length global
    let table_size_global_init = builder.build_constant_uint(usize_type, functions.len() as u64);
    let table_size_global = builder.build_global(
        usize_type,
        "__LUMEN_SYMBOL_TABLE_SIZE",
        Some(table_size_global_init),
    );
    builder.set_alignment(table_size_global, 8);

    // Generate thread local variable for current reduction count
    let i32_type = builder.get_i32_type();
    let reduction_count_init = builder.build_constant_uint(i32_type, 0);
    let reduction_count_global = builder.build_global(
        i32_type,
        "CURRENT_REDUCTION_COUNT",
        Some(reduction_count_init),
    );
    builder.set_thread_local_mode(reduction_count_global, ThreadLocalMode::LocalExec);
    builder.set_linkage(reduction_count_global, Linkage::External);
    builder.set_alignment(reduction_count_global, 8);

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
        &[main_ptr_ty, usize_type, i8ptrptr_type],
        /* varidic= */ false,
    );

    let lang_start_fn_decl = builder.build_external_function(lang_start_symbol_name, lang_start_ty);
    let lang_start_shim_fn = builder.build_external_function(lang_start_alias_name, lang_start_ty);

    let entry_block = builder.build_entry_block(lang_start_shim_fn);
    builder.position_at_end(entry_block);

    let lang_start_args = builder.get_function_params(lang_start_shim_fn);
    let lang_start_call = builder.build_call(lang_start_fn_decl, &lang_start_args, None);
    builder.set_is_tail(lang_start_call, true);
    builder.build_return(lang_start_call);

    // Finalize module
    let module = builder.finish()?;

    // Open ll file for writing
    let ir_path = output_dir.join(&format!("{}.ll", NAME));
    let mut file = File::create(ir_path.as_path())?;
    // Emit IR file
    module.emit_ir(&mut file)?;

    // Open object file for writing
    let obj_path = output_dir.join(&format!("{}.o", NAME));
    let mut file = File::create(obj_path.as_path())?;
    // Emit object file
    module.emit_obj(&mut file)?;

    Ok(Arc::new(CompiledModule::new(
        NAME.to_string(),
        Some(obj_path),
        None,
    )))
}

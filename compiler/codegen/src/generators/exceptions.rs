use std::fs::File;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use liblumen_llvm as llvm;
use liblumen_llvm::builder::ModuleBuilder;
use liblumen_llvm::ir::*;
use liblumen_llvm::target::TargetMachine;
use liblumen_session::{Input, Options, OutputType};

use crate::meta::CompiledModule;

/// Generates an LLVM module containing the top-level exception handler for processes.
pub fn generate(
    options: &Options,
    context: &Context,
    target_machine: TargetMachine,
) -> anyhow::Result<Arc<CompiledModule>> {
    if options.target.arch != "wasm32" {
        generate_standard(options, context, target_machine)
    } else {
        generate_wasm32(options, context, target_machine)
    }
}

fn generate_standard(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
) -> anyhow::Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_exceptions";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Define LLVM types used during generation
    let usize_type = builder.get_usize_type();
    let i8_type = builder.get_i8_type();
    let i8_ptr_type = builder.get_pointer_type(i8_type);
    let i32_type = builder.get_i32_type();
    let fn_type = builder.get_erlang_function_type(1);
    let fn_ptr_type = builder.get_pointer_type(fn_type);
    let void_type = builder.get_void_type();
    let exception_type = builder.get_struct_type(Some("lumen.exception"), &[i8_ptr_type, i32_type]);
    // Matches the layout of ErlangException in liblumen_alloc
    let erlang_error_type = builder.get_struct_type(
        Some("erlang.exception"),
        &[usize_type, usize_type, usize_type, i8_ptr_type, i8_ptr_type],
    );
    let erlang_error_type_ptr = builder.get_pointer_type(erlang_error_type);

    // Define external functions called by the exception handler

    // Personality function used for exception handling
    let personality_fun_ty = builder.get_function_type(i32_type, &[], /* variadic */ true);
    let personality_fun =
        builder.build_external_function("lumen_eh_personality", personality_fun_ty);

    // Define __lumen_start_panic symbol
    let func_ty = builder.get_function_type(void_type, &[i8_ptr_type], /* variadic= */ false);
    let start_panic_fun =
        builder.build_function_with_attrs("__lumen_start_panic", func_ty, Linkage::External, &[]);

    // Define __lumen_panic, which tail calls __lumen_start_panic in
    // order to panic using the right personality function
    let func_ty = builder.get_function_type(void_type, &[i8_ptr_type], /* variadic= */ false);
    let func = builder.build_function_with_attrs("__lumen_panic", func_ty, Linkage::External, &[]);
    func.set_personality(personality_fun);

    let entry_block = builder.build_entry_block(func);
    builder.position_at_end(entry_block);

    let exception_ptr = builder.get_function_param(func, 0);
    let call = builder.build_call(start_panic_fun, &[exception_ptr], None);
    builder.set_is_tail(call, true);
    builder.build_unreachable();

    // Function to extract the Erlang exception term from the raw exception object
    let get_exception_fun_ty = builder.get_function_type(
        erlang_error_type_ptr,
        &[i8_ptr_type],
        /* variadic */ false,
    );
    let get_exception_fun = builder.build_function_with_attrs(
        "__lumen_get_exception",
        get_exception_fun_ty,
        Linkage::External,
        &[
            Attribute::NoUnwind,
            Attribute::ReadOnly,
            Attribute::ArgMemOnly,
        ],
    );

    // Process exit
    let exit_fun_ty = builder.get_function_type(
        void_type,
        &[erlang_error_type_ptr],
        /* variadic */ false,
    );
    let exit_fun = builder.build_function_with_attrs(
        // This cannot be `erlang:exit/1` because this exit is called once the `process_raise`
        // exception has been caught, so it does `Process::exit` directly and yields to the
        // scheduler instead.
        "__lumen_builtin_exit",
        exit_fun_ty,
        Linkage::External,
        &[Attribute::NoReturn],
    );

    // Define global that holds the "type" of Erlang exceptions
    let type_desc = if options.target.options.is_like_msvc {
        let type_info_vtable = builder.declare_global("??_7type_info@@6B@", i8_ptr_type);
        let type_name = builder.build_constant_bytes(b"lumen_panic\0");
        let type_info = builder.build_constant_unnamed_struct(&[
            type_info_vtable,
            builder.build_constant_null(i8_ptr_type),
            type_name,
        ]);
        let type_desc =
            builder.declare_global("__lumen_erlang_error_type_info", builder.type_of(type_info));
        builder.set_linkage(type_desc, Linkage::LinkOnceODR);
        builder.set_unique_comdat(type_desc);
        builder.set_initializer(type_desc, type_info);

        type_desc
    } else {
        builder.build_constant_null(i8_ptr_type)
    };

    // Define exception handler

    let func_ty = builder.get_function_type(
        void_type,
        &[fn_ptr_type, usize_type],
        /* variadic */ false,
    );
    let func = builder.build_external_function("__lumen_trap_exceptions", func_ty);
    func.set_personality(personality_fun);

    // Define all the blocks first so we can reference them
    let entry_block = builder.build_entry_block(func);
    let catch_block = builder.build_named_block(func, "catch");
    let exit_block = builder.build_named_block(func, "exit");

    // Define the entry block
    builder.position_at_end(entry_block);

    let null_erlang_error = builder.build_constant_null(erlang_error_type_ptr);

    // Invoke the `init` function pointer
    let init_fn_ptr = builder.get_function_param(func, 0);
    let init_fn_env_arg = builder.get_function_param(func, 1);
    builder.build_invoke(
        init_fn_ptr,
        &[init_fn_env_arg],
        exit_block,
        catch_block,
        None,
    );

    // Catch error and attempt to handle it
    builder.position_at_end(catch_block);

    // Build landing pad for exception with a single clause for our type info
    let catch_type = if options.target.options.is_like_msvc {
        builder.build_bitcast(type_desc, i8_ptr_type)
    } else {
        type_desc
    };
    let obj = builder.build_landingpad(exception_type, &[catch_type]);

    // Extract the exception object (we ignore the selector)
    let exception_ptr = builder.build_extractvalue(obj, 0);

    // Our personality function ensures that we only enter the landing pad
    // if this is an Erlang exception, so we proceed directly to handling the
    // error, which is a pointer to an ErlangException
    let erlang_error = builder.build_call(get_exception_fun, &[exception_ptr], None);
    builder.build_br(exit_block);

    // Exit the process
    builder.position_at_end(exit_block);

    let exit_value = builder.build_phi(
        erlang_error_type_ptr,
        &[
            (null_erlang_error, entry_block),
            (erlang_error, catch_block),
        ],
    );

    let exit_fun_call = builder.build_call(exit_fun, &[exit_value], None);
    builder.set_is_tail(exit_fun_call, true);
    builder.build_return(ptr::null_mut());

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
        module.emit_asm(&mut file)?;
    }

    // Emit object file
    let obj_path = if let Some(obj_path) = options.maybe_emit(&input, OutputType::Object) {
        let mut file = File::create(obj_path.as_path())?;
        module.emit_obj(&mut file)?;
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

fn generate_wasm32(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
) -> anyhow::Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_exceptions";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Define LLVM types used during generation
    let usize_type = builder.get_usize_type();
    let i8_type = builder.get_i8_type();
    let i8_ptr_type = builder.get_pointer_type(i8_type);
    let i32_type = builder.get_i32_type();
    let fn_type = builder.get_erlang_function_type(1);
    let fn_ptr_type = builder.get_pointer_type(fn_type);
    let void_type = builder.get_void_type();
    let _exception_type =
        builder.get_struct_type(Some("lumen.exception"), &[i8_ptr_type, i32_type]);
    let erlang_error_type = builder.get_struct_type(
        Some("erlang.exception"),
        &[usize_type, usize_type, usize_type, i8_ptr_type, i8_ptr_type],
    );
    let erlang_error_type_ptr = builder.get_pointer_type(erlang_error_type);

    // Define external functions called by the exception handler

    // Personality function used for exception handling
    let personality_fun_ty = builder.get_function_type(i32_type, &[], /* variadic */ true);
    let personality_fun =
        builder.build_external_function("lumen_eh_personality", personality_fun_ty);

    // Define __lumen_start_panic symbol
    let func_ty = builder.get_function_type(void_type, &[i8_ptr_type], /* variadic= */ false);
    let start_panic_fun = builder.build_function_with_attrs(
        "__lumen_start_panic",
        func_ty,
        Linkage::External,
        &[Attribute::NoReturn],
    );

    // Define __lumen_panic, which tail calls __lumen_start_panic in
    // order to panic using the right personality function
    let func_ty = builder.get_function_type(void_type, &[i8_ptr_type], /* variadic= */ false);
    let func = builder.build_function_with_attrs(
        "__lumen_panic",
        func_ty,
        Linkage::External,
        &[Attribute::NoReturn],
    );
    func.set_personality(personality_fun);

    let entry_block = builder.build_entry_block(func);
    builder.position_at_end(entry_block);

    let exception_ptr = builder.get_function_param(func, 0);
    let call = builder.build_call(start_panic_fun, &[exception_ptr], None);
    builder.set_is_tail(call, true);
    builder.build_unreachable();

    // Function to extract the Erlang exception term from the raw exception object
    let get_exception_fun_ty = builder.get_function_type(
        usize_type,
        &[erlang_error_type_ptr],
        /* variadic */ false,
    );
    let get_exception_fun = builder.build_function_with_attrs(
        "__lumen_get_exception",
        get_exception_fun_ty,
        Linkage::External,
        &[
            Attribute::NoUnwind,
            Attribute::ReadOnly,
            Attribute::ArgMemOnly,
        ],
    );

    // Process exit
    let exit_fun_ty = builder.get_function_type(
        void_type,
        &[erlang_error_type_ptr],
        /* variadic */ false,
    );
    let exit_fun = builder.build_function_with_attrs(
        // This cannot be `erlang:exit/1` because this exit is called once the `process_raise`
        // exception has been caught, so it does `Process::exit` directly and yields to the
        // scheduler instead.
        "__lumen_builtin_exit",
        exit_fun_ty,
        Linkage::External,
        &[Attribute::NoReturn],
    );

    // Define exception handler

    let func_ty = builder.get_function_type(
        void_type,
        &[fn_ptr_type, usize_type],
        /* variadic */ false,
    );
    let func = builder.build_external_function("__lumen_trap_exceptions", func_ty);
    func.set_personality(personality_fun);

    // Define all the blocks first so we can reference them
    let entry_block = builder.build_entry_block(func);
    let catch_dispatch_block = builder.build_named_block(func, "catch.dispatch");
    let catch_start_block = builder.build_named_block(func, "catch.start");
    let catch_block = builder.build_named_block(func, "catch");
    let catchret_block = builder.build_named_block(func, "catch.end");
    let exit_block = builder.build_named_block(func, "exit");

    // Define the entry block
    builder.position_at_end(entry_block);

    let null_erlang_error = builder.build_constant_null(erlang_error_type_ptr);
    let exit_value = builder.build_alloca(erlang_error_type_ptr);
    builder.build_store(exit_value, null_erlang_error);

    // Invoke the `init` function pointer
    let init_fn_ptr = builder.get_function_param(func, 0);
    let init_fn_env_arg = builder.get_function_param(func, 1);
    builder.build_invoke(
        init_fn_ptr,
        &[init_fn_env_arg],
        exit_block,
        catch_block,
        None,
    );

    // Define catch entry
    // %1 = catchswitch within none [label %catch.start] unwind to caller
    builder.position_at_end(catch_dispatch_block);
    let catchswitch = builder.build_catchswitch(None, None, &[catch_start_block]);

    // Define catch landing pad
    // %2 = catchpad within %1 [i8* null]
    // %3 = call i8* @llvm.wasm.get.exception(token %2)
    // %4 = call i32 @llvm.wasm.get.ehselector(token %2)
    // br label %catch
    builder.position_at_end(catch_start_block);
    let catchpad = builder.build_catchpad(
        Some(catchswitch),
        &[builder.build_constant_null(i8_ptr_type)],
    );
    let get_exception = builder.get_intrinsic("llvm.wasm.get.exception");
    let exception_ptr = builder.build_call(get_exception, &[catchpad.pad()], None);
    // We ignore the selector, but would access it like so if needed:
    // let get_selector = builder.get_intrinsic("llvm.wasm.get.ehselector");
    // let selector = builder.build_call(get_selector, &[catchpad.pad()], None);
    builder.build_br(catch_block);

    // Our personality function ensures that we only enter the landing pad
    // if this is an Erlang exception, so we proceed directly to handling the
    // error, a pointer to an ErlangException
    let erlang_error = builder.build_call(get_exception_fun, &[exception_ptr], Some(&catchpad));
    builder.build_store(exit_value, erlang_error);

    // That's it, return the caught error
    builder.build_br(catchret_block);

    // Finish catch handling
    builder.position_at_end(catchret_block);
    builder.build_catchret(&catchpad, Some(exit_block));

    // Exit the process
    builder.position_at_end(exit_block);

    let exit_fun_call = builder.build_call(exit_fun, &[exit_value], None);
    builder.set_is_tail(exit_fun_call, true);
    builder.build_return(ptr::null_mut());

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
        module.emit_asm(&mut file)?;
    }

    // Emit object file
    let obj_path = if let Some(obj_path) = options.maybe_emit(&input, OutputType::Object) {
        let mut file = File::create(obj_path.as_path())?;
        module.emit_obj(&mut file)?;
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

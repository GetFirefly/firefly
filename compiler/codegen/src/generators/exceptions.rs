use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::mem;
use std::path::Path;
use std::ptr;
use std::sync::Arc;

use libeir_intern::{Ident, Symbol};
use libeir_ir::FunctionIdent;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_llvm as llvm;
use liblumen_llvm::attributes::{Attribute, AttributePlace};
use liblumen_llvm::builder::{ICmp, ModuleBuilder};
use liblumen_llvm::enums::{Linkage, ThreadLocalMode};
use liblumen_llvm::target::TargetMachine;
use liblumen_session::Options;
use liblumen_term::{
    Encoding, Encoding32, Encoding64, Encoding64Nanboxed, EncodingType, Tag, TermKind,
};

use crate::meta::CompiledModule;
use crate::Result;

/// Generates an LLVM module containing the top-level exception handler for processes.
pub fn generate(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
    output_dir: &Path,
) -> Result<Arc<CompiledModule>> {
    if options.target.arch != "wasm32" {
        generate_standard(options, context, target_machine, output_dir)
    } else {
        generate_wasm32(options, context, target_machine, output_dir)
    }
}

fn generate_standard(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
    output_dir: &Path,
) -> Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_exceptions";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Define LLVM types used during generation
    let usize_type = builder.get_usize_type();
    let i1_type = builder.get_integer_type(1);
    let i8_type = builder.get_i8_type();
    let i8_ptr_type = builder.get_pointer_type(i8_type);
    let i32_type = builder.get_i32_type();
    let fn_type = builder.get_erlang_function_type(0);
    let fn_ptr_type = builder.get_pointer_type(fn_type);
    let void_type = builder.get_void_type();
    let exception_type = builder.get_struct_type(Some("lumen.exception"), &[i8_ptr_type, i32_type]);
    let erlang_error_type = builder.get_struct_type(
        Some("tuple3"),
        &[usize_type, usize_type, usize_type, usize_type],
    );
    let erlang_error_type_ptr = builder.get_pointer_type(erlang_error_type);

    // Define external functions called by the exception handler

    // Personality function used for exception handling
    let personality_fun_ty = builder.get_function_type(i32_type, &[], /* variadic */ true);
    let personality_fun =
        builder.build_external_function("lumen_eh_personality", personality_fun_ty);

    // Function to extract the Erlang exception term from the raw exception object
    let get_exception_fun_ty =
        builder.get_function_type(usize_type, &[i8_ptr_type], /* variadic */ false);
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

    // Term comparison
    let cmp_eq_fun_ty = builder.get_function_type(
        i1_type,
        &[usize_type, usize_type],
        /* variadic */ false,
    );
    let cmp_eq_fun = builder.build_function_with_attrs(
        "__lumen_builtin_cmp.eq",
        cmp_eq_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind],
    );

    // Process heap allocation
    let malloc_fun_ty = builder.get_function_type(
        i8_ptr_type,
        &[i32_type, usize_type],
        /* variadic */ false,
    );
    let malloc_fun = builder.build_function_with_attrs(
        "__lumen_builtin_malloc",
        malloc_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind],
    );

    // Process exit
    let exit_fun_ty =
        builder.get_function_type(void_type, &[usize_type], /* variadic */ false);
    let exit_fun = builder.build_function_with_attrs(
        "__lumen_builtin_exit",
        exit_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind, Attribute::NoReturn],
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

    let func_ty = builder.get_function_type(void_type, &[fn_ptr_type], /* variadic */ false);
    let func = builder.build_external_function("__lumen_trap_exceptions", func_ty);
    builder.set_personality(func, personality_fun);

    // Define all the blocks first so we can reference them
    let entry_block = builder.build_entry_block(func);
    let catch_block = builder.build_named_block(func, "catch");
    let handle_throw_block = builder.build_named_block(func, "handle.throw");
    let handle_throw_finish_block = builder.build_named_block(func, "handle.throw.finish");
    let caught_block = builder.build_named_block(func, "caught");
    let resume_block = builder.build_named_block(func, "resume");
    let exit_block = builder.build_named_block(func, "exit");

    // Define the entry block
    builder.position_at_end(entry_block);

    // Allocate all constants here and reuse them as needed
    let throw = Symbol::intern("throw");
    let nocatch = Symbol::intern("nocatch");

    let throw_atom = build_constant_atom(&builder, throw.as_usize(), options);
    let nocatch_atom = build_constant_atom(&builder, nocatch.as_usize(), options);
    let nocatch_header = build_tuple_header(&builder, 2, options);
    let box_tag = build_constant_box_tag(&builder, usize_type, options);
    let tuple_kind = builder.build_constant_uint(i32_type, TermKind::Tuple as u64);
    let nocatch_arity = builder.build_constant_uint(usize_type, 2);
    let updated_error_header = build_tuple_header(&builder, 2, options);

    // Invoke the `init` function pointer
    let init_fn_ptr = builder.get_function_param(func, 0);
    let invoke_init = builder.build_invoke(init_fn_ptr, &[], exit_block, catch_block, None);

    // Catch error and attempt to handle it
    builder.position_at_end(catch_block);

    // Build landing pad for exception with a single clause for our type info
    let catch_type = if options.target.options.is_like_msvc {
        builder.build_bitcast(type_desc, i8_ptr_type)
    } else {
        type_desc
    };
    let obj = builder.build_landingpad(exception_type, personality_fun, &[catch_type]);

    // Extract the exception object (we ignore the selector)
    let exception_ptr = builder.build_extractvalue(obj, 0);

    // Our personality function ensures that we only enter the landing pad
    // if this is an Erlang exception, so we proceed directly to handling the
    // error, a boxed 3-tuple
    let error_box = builder.build_call(get_exception_fun, &[exception_ptr], None);

    // Drop the last element of the tuple and make the exit value just `{kind, reason}`,
    // all we need to do for this is to rewrite the arity value in the tuple header, which
    // can be done by writing to the pointer we just cast from
    let error_ptr = match options.target.options.encoding {
        // In this encoding scheme, boxes are always pointers
        EncodingType::Encoding64Nanboxed => {
            builder.build_inttoptr(error_box, erlang_error_type_ptr)
        }
        // For all other encoding schemes, we unmask the pointer first
        _ => {
            let box_tag = box_tag.unwrap();
            let untagged = builder.build_and(error_box, builder.build_not(box_tag));
            builder.build_inttoptr(untagged, erlang_error_type_ptr)
        }
    };
    let error_kind_ptr = builder.build_struct_gep(error_ptr, 1);
    let error_kind = builder.build_load(usize_type, error_kind_ptr);

    let is_throw = builder.build_call(cmp_eq_fun, &[error_kind, throw_atom], None);
    builder.build_condbr(is_throw, handle_throw_block, caught_block);

    // Try to allocate memory on the process heap for the {nocatch, Reason} tuple
    builder.position_at_end(handle_throw_block);

    let nocatch_tuple_opaque = builder.build_call(malloc_fun, &[tuple_kind, nocatch_arity], None);
    builder.build_condbr(
        builder.build_is_null(nocatch_tuple_opaque),
        handle_throw_finish_block,
        caught_block,
    );

    // Handle translating throw to error with {nocatch, Reason} reason
    builder.position_at_end(handle_throw_finish_block);

    // Write the tuple elements
    let nocatch_tuple = builder.build_bitcast(nocatch_tuple_opaque, erlang_error_type_ptr);

    let nocatch_header_ptr = builder.build_struct_gep(nocatch_tuple, 0);
    builder.build_store(nocatch_header, nocatch_header_ptr);
    let nocatch_kind_ptr = builder.build_struct_gep(nocatch_tuple, 1);
    builder.build_store(nocatch_atom, nocatch_kind_ptr);

    let error_reason_ptr = builder.build_struct_gep(error_ptr, 2);
    let error_reason = builder.build_load(usize_type, error_reason_ptr);
    let nocatch_reason_ptr = builder.build_struct_gep(nocatch_tuple, 2);
    builder.build_store(error_reason, nocatch_reason_ptr);

    let nocatch_box = match options.target.options.encoding {
        EncodingType::Encoding64Nanboxed => builder.build_ptrtoint(nocatch_tuple, usize_type),
        _ => {
            let box_tag = box_tag.unwrap();
            let boxed = builder.build_ptrtoint(nocatch_tuple, usize_type);
            builder.build_or(boxed, box_tag)
        }
    };

    // Update the original error kind and reason
    builder.build_store(throw_atom, error_kind_ptr);
    builder.build_store(nocatch_box, error_reason_ptr);

    builder.build_br(caught_block);

    // Finish handling the error by forcefully resizing the tuple to drop the last element
    builder.position_at_end(caught_block);

    let error_header_ptr = builder.build_struct_gep(error_ptr, 0);
    builder.build_store(updated_error_header, error_header_ptr);

    builder.build_br(exit_block);

    // Exit the process
    builder.position_at_end(exit_block);

    let exit_value = builder.build_phi(
        usize_type,
        &[(invoke_init, entry_block), (error_box, caught_block)],
    );

    let exit_fun_call = builder.build_call(exit_fun, &[exit_value], None);
    builder.set_is_tail(exit_fun_call, true);
    builder.build_return(ptr::null_mut());

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

fn generate_wasm32(
    options: &Options,
    context: &llvm::Context,
    target_machine: &TargetMachine,
    output_dir: &Path,
) -> Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_exceptions";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Define LLVM types used during generation
    let usize_type = builder.get_usize_type();
    let i1_type = builder.get_integer_type(1);
    let i8_type = builder.get_i8_type();
    let i8_ptr_type = builder.get_pointer_type(i8_type);
    let i32_type = builder.get_i32_type();
    let fn_type = builder.get_erlang_function_type(0);
    let fn_ptr_type = builder.get_pointer_type(fn_type);
    let void_type = builder.get_void_type();
    let exception_type = builder.get_struct_type(Some("lumen.exception"), &[i8_ptr_type, i32_type]);
    let erlang_error_type = builder.get_struct_type(
        Some("tuple3"),
        &[usize_type, usize_type, usize_type, usize_type],
    );
    let erlang_error_type_ptr = builder.get_pointer_type(erlang_error_type);

    // Define external functions called by the exception handler

    // Personality function used for exception handling
    let personality_fun_ty = builder.get_function_type(i32_type, &[], /* variadic */ true);
    let personality_fun =
        builder.build_external_function("lumen_eh_personality", personality_fun_ty);

    // Function to extract the Erlang exception term from the raw exception object
    let get_exception_fun_ty =
        builder.get_function_type(usize_type, &[i8_ptr_type], /* variadic */ false);
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

    // Term comparison
    let cmp_eq_fun_ty = builder.get_function_type(
        i1_type,
        &[usize_type, usize_type],
        /* variadic */ false,
    );
    let cmp_eq_fun = builder.build_function_with_attrs(
        "__lumen_builtin_cmp.eq",
        cmp_eq_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind],
    );

    // Process heap allocation
    let malloc_fun_ty = builder.get_function_type(
        i8_ptr_type,
        &[i32_type, usize_type],
        /* variadic */ false,
    );
    let malloc_fun = builder.build_function_with_attrs(
        "__lumen_builtin_malloc",
        malloc_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind],
    );

    // Process exit
    let exit_fun_ty =
        builder.get_function_type(void_type, &[usize_type], /* variadic */ false);
    let exit_fun = builder.build_function_with_attrs(
        "__lumen_builtin_exit",
        exit_fun_ty,
        Linkage::External,
        &[Attribute::NoUnwind, Attribute::NoReturn],
    );

    // Define exception handler

    let func_ty = builder.get_function_type(void_type, &[fn_ptr_type], /* variadic */ false);
    let func = builder.build_external_function("__lumen_trap_exceptions", func_ty);
    builder.set_personality(func, personality_fun);

    // Define all the blocks first so we can reference them
    let entry_block = builder.build_entry_block(func);
    let catch_dispatch_block = builder.build_named_block(func, "catch.dispatch");
    let catch_start_block = builder.build_named_block(func, "catch.start");
    let catch_block = builder.build_named_block(func, "catch");
    let handle_throw_block = builder.build_named_block(func, "handle.throw");
    let handle_throw_finish_block = builder.build_named_block(func, "handle.throw.end");
    let caught_block = builder.build_named_block(func, "caught");
    let catchret_block = builder.build_named_block(func, "catch.end");
    let exit_block = builder.build_named_block(func, "exit");

    // Define the entry block
    builder.position_at_end(entry_block);

    let exit_value = builder.build_alloca(usize_type);

    // Allocate all constants here and reuse them as needed
    let throw = Symbol::intern("throw");
    let nocatch = Symbol::intern("nocatch");

    let throw_atom = build_constant_atom(&builder, throw.as_usize(), options);
    let nocatch_atom = build_constant_atom(&builder, nocatch.as_usize(), options);
    let nocatch_header = build_tuple_header(&builder, 2, options);
    let box_tag = build_constant_box_tag(&builder, usize_type, options);
    let tuple_kind = builder.build_constant_uint(i32_type, TermKind::Tuple as u64);
    let nocatch_arity = builder.build_constant_uint(usize_type, 2);
    let updated_error_header = build_tuple_header(&builder, 2, options);

    // Invoke the `init` function pointer
    let init_fn_ptr = builder.get_function_param(func, 0);
    let invoke_init = builder.build_invoke(init_fn_ptr, &[], exit_block, catch_block, None);

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
    // error, a boxed 3-tuple
    let error_box = builder.build_call(get_exception_fun, &[exception_ptr], Some(&catchpad));
    builder.build_store(exit_value, error_box);

    // Drop the last element of the tuple and make the exit value just `{kind, reason}`,
    // all we need to do for this is to rewrite the arity value in the tuple header, which
    // can be done by writing to the pointer we just cast from
    let error_ptr = {
        let box_tag = box_tag.unwrap();
        let untagged = builder.build_and(error_box, builder.build_not(box_tag));
        builder.build_inttoptr(untagged, erlang_error_type_ptr)
    };
    let error_kind_ptr = builder.build_struct_gep(error_ptr, 1);
    let error_kind = builder.build_load(usize_type, error_kind_ptr);

    let is_throw = builder.build_call(cmp_eq_fun, &[error_kind, throw_atom], Some(&catchpad));
    builder.build_condbr(is_throw, handle_throw_block, caught_block);

    // Try to allocate memory on the process heap for the {nocatch, Reason} tuple
    builder.position_at_end(handle_throw_block);

    let nocatch_tuple_opaque =
        builder.build_call(malloc_fun, &[tuple_kind, nocatch_arity], Some(&catchpad));
    builder.build_condbr(
        builder.build_is_null(nocatch_tuple_opaque),
        handle_throw_finish_block,
        caught_block,
    );

    // Handle translating throw to error with {nocatch, Reason} reason
    builder.position_at_end(handle_throw_finish_block);

    // Write the tuple elements
    let nocatch_tuple = builder.build_bitcast(nocatch_tuple_opaque, erlang_error_type_ptr);

    let nocatch_header_ptr = builder.build_struct_gep(nocatch_tuple, 0);
    builder.build_store(nocatch_header, nocatch_header_ptr);
    let nocatch_kind_ptr = builder.build_struct_gep(nocatch_tuple, 1);
    builder.build_store(nocatch_atom, nocatch_kind_ptr);

    let error_reason_ptr = builder.build_struct_gep(error_ptr, 2);
    let error_reason = builder.build_load(usize_type, error_reason_ptr);
    let nocatch_reason_ptr = builder.build_struct_gep(nocatch_tuple, 2);
    builder.build_store(error_reason, nocatch_reason_ptr);

    let nocatch_box = {
        let box_tag = box_tag.unwrap();
        let boxed = builder.build_ptrtoint(nocatch_tuple, usize_type);
        builder.build_or(boxed, box_tag)
    };

    // Update the original error kind and reason
    builder.build_store(throw_atom, error_kind_ptr);
    builder.build_store(nocatch_box, error_reason_ptr);

    builder.build_br(caught_block);

    // Finish handling the error by forcefully resizing the tuple to drop the last element
    builder.position_at_end(caught_block);

    let error_header_ptr = builder.build_struct_gep(error_ptr, 0);
    builder.build_store(updated_error_header, error_header_ptr);
    builder.build_br(catchret_block);

    // Finish catch handling
    builder.position_at_end(catchret_block);
    builder.build_catchret(&catchpad, Some(exit_block));

    // Exit the process
    builder.position_at_end(exit_block);

    let ret_value = builder.build_phi(
        usize_type,
        &[(invoke_init, entry_block), (exit_value, catchret_block)],
    );

    let exit_fun_call = builder.build_call(exit_fun, &[ret_value], None);
    builder.set_is_tail(exit_fun_call, true);
    builder.build_return(ptr::null_mut());

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

fn build_constant_box_tag<'a>(
    builder: &'a ModuleBuilder<'a>,
    ty: llvm::Type,
    options: &Options,
) -> Option<llvm::Value> {
    match options.target.options.encoding {
        // In this encoding scheme, boxes are always pointers
        EncodingType::Encoding64Nanboxed => None,
        // For all other encoding schemes, we unmask the pointer first
        EncodingType::Encoding64 => {
            Some(builder.build_constant_uint(ty, Encoding64::TAG_BOXED as u64))
        }
        EncodingType::Encoding32 => {
            Some(builder.build_constant_uint(ty, Encoding32::TAG_BOXED as u64))
        }
        EncodingType::Default if options.target.target_pointer_width == 64 => {
            Some(builder.build_constant_uint(ty, Encoding64::TAG_BOXED as u64))
        }
        EncodingType::Default if options.target.target_pointer_width == 32 => {
            Some(builder.build_constant_uint(ty, Encoding32::TAG_BOXED as u64))
        }
        _ => unreachable!(),
    }
}

fn build_constant_atom<'a>(
    builder: &'a ModuleBuilder<'a>,
    id: usize,
    options: &Options,
) -> llvm::Value {
    let usize_type = builder.get_usize_type();
    match options.target.options.encoding {
        EncodingType::Encoding64Nanboxed => {
            let tagged = Encoding64Nanboxed::encode_immediate_with_tag(id as u64, Tag::Atom);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Encoding64 => {
            let tagged = Encoding64::encode_immediate_with_tag(id as u64, Tag::Atom);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Encoding32 => {
            let tagged = Encoding32::encode_immediate_with_tag(id as u32, Tag::Atom);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Default if options.target.target_pointer_width == 64 => {
            let tagged = Encoding64::encode_immediate_with_tag(id as u64, Tag::Atom);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Default if options.target.target_pointer_width == 32 => {
            let tagged = Encoding32::encode_immediate_with_tag(id as u32, Tag::Atom);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        _ => unreachable!(),
    }
}

fn build_tuple_header<'a>(
    builder: &'a ModuleBuilder<'a>,
    arity: usize,
    options: &Options,
) -> llvm::Value {
    let usize_type = builder.get_usize_type();
    match options.target.options.encoding {
        EncodingType::Encoding64Nanboxed => {
            let tagged = Encoding64Nanboxed::encode_header_with_tag(arity as u64, Tag::Tuple);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Encoding64 => {
            let tagged = Encoding64::encode_header_with_tag(arity as u64, Tag::Tuple);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Encoding32 => {
            let tagged = Encoding32::encode_header_with_tag(arity as u32, Tag::Tuple);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Default if options.target.target_pointer_width == 64 => {
            let tagged = Encoding64::encode_header_with_tag(arity as u64, Tag::Tuple);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        EncodingType::Default if options.target.target_pointer_width == 32 => {
            let tagged = Encoding32::encode_header_with_tag(arity as u32, Tag::Tuple);
            builder.build_constant_uint(usize_type, tagged as u64)
        }
        _ => unreachable!(),
    }
}

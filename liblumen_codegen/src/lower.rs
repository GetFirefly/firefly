#![allow(unused)]
use std::mem;
use std::collections::{HashMap, HashSet};

use num_bigint::BigInt;

use inkwell::AddressSpace;
use inkwell::context::Context;
use inkwell::module::Module as LLVMModule;
use inkwell::module::Linkage;
use inkwell::values::*;
use inkwell::types::{IntType, BasicType, ArrayType};
use inkwell::targets::Target;
use inkwell::builder::Builder;
use inkwell::attributes::{Attribute, AttributeLoc};
use inkwell::basic_block::BasicBlock;

use libeir_ir::{Module, Block, FunctionIdent, Function};
use libeir_ir::{Const, ConstKind};
use libeir_ir::AtomicTerm;

use liblumen_alloc::erts::term::Term;
use liblumen_core::sys::sysconfg::MIN_ALIGN;

use super::{Result, CodeGenError, Config};

/// Lowers a module from EIR to LLVM IR
pub fn module(module: &Module, config: &Config) -> Result<LLVMModule> {
    let context = Context::create();
    let builder = context.create_builder();
    // NOTE: For each function or global emitted, use the most private linkage type possible (private, internal or linkonce_odr preferably).
    // Doing so will make LLVM’s inter-procedural optimizations much more effective.
    let llvm_module = context.create_module(module.name.as_str().get());
    llvm_module.set_source_file_name("nofile");
    llvm_module.set_target(config.target());

    let target_data = config.target_data();
    let data_layout = target_data.get_data_layout();
    llvm_module.set_data_layout(&data_layout);

    let int_ty = target_data.ptr_sized_int_type_in_context(&context, None);

    // Declare intrinsics
    self::declare_intrinsics(&context, &llvm_module, int_ty.clone());
    // Declare builtins
    self::declare_builtins(&context, &llvm_module, &builder, int_ty.clone());

    // Define module functions
    for (fun_ident, fun) in &module.functions {
        assert_eq!(fun_ident.module, module.name, "expected function and module to have the same module name!");
        self::function(fun_ident, fun, int_ty, &context, &llvm_module, &builder)?;
    }

    Ok(llvm_module)
}

// Required intrinsics:
// - declare ptrty llvm.ptrmask(ptrty %ptr, intty %mask) readnone speculatable
fn declare_intrinsics(ctx: &Context, module: &LLVMModule, int_ty: IntType) {
    let ptr_ty = int_ty.ptr_type(AddressSpace::Generic);
    module.get_intrinsic("llvm.ptrmask", &[ptr_ty.into(), int_ty.into()]);
}

// Core builtins:
// - lumen.intrinsics.unbox: given a tagged pointer, strips the tag off leaving a usable pointer
const UNBOX_FN: &'static str = "lumen.intrinsics.unbox";
// The primary tag is given by masking bits 0-2
const MASK_PRIMARY_64: usize = 0xC000_0000_0000_0000;
// The literal tag is given by masking the lowest bit
const MASK_LITERAL_64: usize = 0x1;
// The primary tag is given by masking bits 0-1
const MASK_PRIMARY_32: usize = 0b11;
// The literal flag is the highest bit.  Assert check that we're not hitting it under large
// memory usage.
const MASK_LITERAL_32: usize = 1 << (32 - 1);

fn declare_builtins(ctx: &Context,
                    module: &LLVMModule,
                    builder: &Builder,
                    int_ty: IntType) {
    let ptr_ty = ctx.i8_type().ptr_type(AddressSpace::Generic);
    let term_ptr_ty = int_ty.ptr_type(AddressSpace::Generic);
    let void_ty = ctx.void_type();
    let nounwind_attr_kind = Attribute::get_named_enum_kind_id("nounwind");
    let nounwind_attr = ctx.create_enum_attribute(nounwind_attr_kind, 1);
    let readnone_attr_kind = Attribute::get_named_enum_kind_id("readnone");
    let readnone_attr = ctx.create_enum_attribute(readnone_attr_kind, 1);
    let speculatable_attr_kind = Attribute::get_named_enum_kind_id("speculatable");
    let speculatable_attr = ctx.create_enum_attribute(speculatable_attr_kind, 1);
    let alwaysinline_attr_kind = Attribute::get_named_enum_kind_id("alwaysinline");
    let alwaysinline_attr = ctx.create_enum_attribute(alwaysinline_attr_kind, 1);
    // llvm.intrinsics.unbox(*mut usize) -> *mut usize;
    let unbox_fn_ty = term_ptr_ty.fn_type(&[term_ptr_ty.clone().into()], /*varargs*/false);
    let unbox_fn = module.add_function("lumen.intrinsics.unbox", unbox_fn_ty, None);
    unbox_fn.add_attribute(AttributeLoc::Function, nounwind_attr.clone());
    unbox_fn.add_attribute(AttributeLoc::Function, readnone_attr.clone());
    unbox_fn.add_attribute(AttributeLoc::Function, speculatable_attr.clone());
    unbox_fn.add_attribute(AttributeLoc::Function, alwaysinline_attr.clone());
    let boxed_ptr_val = unbox_fn.get_first_param().unwrap();
    let entry = unbox_fn.append_basic_block("entry");
    builder.position_at_end(&entry);
    let ptrmask = module.get_intrinsic("llvm.ptrmask", &[term_ptr_ty.into(), int_ty.into()]);
    let mask = match int_ty.get_bit_width() {
        32 => MASK_PRIMARY_32 | MASK_LITERAL_32,
        64 => MASK_PRIMARY_64 | MASK_LITERAL_64,
        bw => panic!("unsupported target bit width ({})!", bw)
    };
    let mask_val = int_ty.const_int(mask, /*signextend*/false);
    let ret_val = builder.build_call(ptrmask, &[boxed_ptr_val, mask_val], "llvm.unbox")
                         .set_tail_call(true)
                         .try_as_basic_value()
                         .left()
                         .unwrap();
    builder.build_return(Some(&ret_val));
}

/// Lowers a function from EIR to LLVM IR in the provided module.
///
/// The name/arity and body of the function are the EIR pieces provided.
///
/// This function receives the current LLVM module and context, from which types and builders can be constructed
fn function(name: &FunctionIdent,
            fun: &Function,
            int_ty: IntType,
            ctx: &Context,
            module: &LLVMModule,
            builder: &Builder) -> Result<()> {
    use inkwell::types::*;

    // # Define common types
    // i8*
    let i8ptr_ty = ctx.i8_type().ptr_type(AddressSpace::Generic);
    // u32*/u64*, i.e. Term*
    let term_ptr_ty = int_ty.ptr_type(AddressSpace::Generic);

    // Verify function arguments
    let entry_argc = fun.entry_arg_num();
    assert!(entry_argc >= 2);
    assert_eq!(name.arity, entry_argc - 2, "function arity and entry block argument count don't match!");
    // Build argument type list for function signature
    let entry_block = fun.block_entry();
    let mut entry_args = Vec::with_capacity(entry_argc);
    entry_args.push(cont_ty.clone());
    entry_args.push(escape_cont_ty.clone());
    for entry_arg in fun.block_args(entry_block)[2..] {
        entry_args.push(int_ty.clone());
    }
    // Build function type
    let fun_ty = int_ty.fn_type(entry_args.as_slice(), /*varargs*/false);

    // Lift all constants into the global constant pool
    for val in fun.iter_constants() {
        let constant = fun.value_const(val).unwrap();
        let constant_kind = fun.constant_container.const_kind(constant);
        define_constant(constant, constant_value, constant_kind, int_ty, ctx, module);
    }

    // Extract the lowering metadata for the function
    let lowering = libeir_lowerutils::analyze(fun);
    assert!(lowering.block_modes[entry_block].is_fun(), "entry block must be a function block");

    let block_graph = fun.block_graph();

    // Find the functions we need to generate/closures we need to lift
    let mut funs: HashSet<Block> = HashSet::new();
    funs.insert(entry_block);
    let mut fun_scopes: HashMap<Block, HashSet<Block>> = HashMap::new();
    while Some(scope_entry) = funs.difference(&fun_scopes.keys().collect()).iter().next() {
        let mut walked = HashSet::new();
        let mut scope = HashSet::new();
        scope.insert(*scope_entry);

        while Some(scope_block) = scope.difference(&walked).iter().next() {
            walked.insert(*scope_block);
            for target in block_graph.outgoing(*scope_block) {
                match lowering.block_modes[target] {
                    BlockMode::BasicBlock => scope.insert(target),
                    BlockMode::Function { .. } => funs.insert(target),
                }
            }
        }

        fun_scopes[scope_entry] = scope;
    }

    for (entry, scope) in fun_scopes.iter() {
        let name = if entry == entry_block {
            name
        } else {
            &FunctionIdent{
                module: name.module.clone(),
                name: Ident::from_str(&format!("{}-fun-{}", name.name, name.arity)),
                arity: name.arity,
            }
        };
        self::generate_function(
            name,
            entry,
            scope,
            &lowering,
            int_ty,
            ctx,
            module,
            builder
        );
    }
}

fn generate_function(name: &FunctionIdent,
                     fun: &Function,
                     entry: Block,
                     scope: &HashSet<Block>,
                     lowering: &LowerData,
                     int_ty: IntType,
                     ctx: &Context,
                     module: &LLVMModule,
                     builder: &Builder) {
    // Types
    // cont(Term) -> Term;
    let cont_ty = int_ty.fn_type(&[int_ty.clone().into()]);
    // escape(Term, Term, Term) -> Term;
    let escape_cont_ty = int_ty.fn_type(&[int_ty.clone().into(), int_ty.clone().into(), int_ty.clone().into()]);

    // Generates function names like `define @"module:function/arity"(...)` or `define @"module:function-fun-N/arity"(...)`
    let function_name = format!("\"{}\"", name);

    // Build argument type list for function signature
    let entry_argc = fun.block_args(entry).len();
    let mut entry_args = Vec::with_capacity(entry_argc);
    assert!(entry_argc >= 2);
    assert_eq!(name.arity, entry_argc - 2, "function arity and entry block argument count don't match!");
    entry_args.push(cont_ty.clone());
    entry_args.push(escape_cont_ty.clone());
    for entry_arg in fun.block_args(entry_block)[2..] {
        entry_args.push(int_ty.clone());
    }

    // Build function type
    let fun_ty = int_ty.fn_type(entry_args.as_slice(), /*varargs*/false);

    // Define the function entry point
    let f = module.add_function(&function_name, fun_ty, None);
    let init_bb = f.append_basic_block("entry");
    builder.position_at_end(&init_bb);

    // TODO: Generate stack frame

    // If this is a closure, extract the environment
    // A closure will have more than 1 live value, otherwise
    // it is a regular function
    let live = lowering.live.live[&entry];
    if live.size(&lowering.live.pool) > 0 {
        panic!("closure env unpacking is unimplemented!");
    }

    // Make the init block branch to the EIR entry block
    let entry_bb = f.append_basic_block(&format!("{}", entry));
    builder.position_at_end(&init_bb);
    builder.build_unconditional_branch(&entry_bb);

    // Construct basic block map for future reference
    let bbs: HashMap<_, _> = scope.iter()
                   .map(|block| (block, f.append_basic_block(&format!("{}", block))))
                   .collect();

    // Lower entry block
    match fun.block_kind(entry_block).unwrap() {
        OpKind::Call
        | OpKind::CaptureFunction
        | OpKind::IfBool
        | OpKind::Intrinsic
        | OpKind::Match { .. } => lower_block(),
        | OpKind::Unreachable => {
            // Nothing more to do, so return early here
            builder.build_unreachable();
            return;
        }
        op => {
            panic!("unexpected operation in entry block of {}: {:?}", &function_name, op);
        }
    }

    // Lower successor blocks
    for (block, bb) in bbs.iter() {
        lower_block(&block, &bb)
    }
}

fn lower_block(block: &Block,
               bb: &BasicBlock,
               ctx: &Context,
               module: &LLVMModule,
               builder: &Builder) {
    // Lower primops
    // Lower operation
    let args = fun.block_args(block);
    let reads = fun.block_reads(block);
    match fun.block_kind(block).unwrap() {
        // i8*, Term...
        OpKind::Call => {
            let fun = reads[0];
            // Build arguments
            let args = reads[1..].iter().map(|r| {
                match fun.value_kind(*r) {
                    ValueKind::Block(b) => {
                        // A closure env
                    }
                    ValueKind::Argument(_, _) => {
                        // The value itself
                    }
                    ValueKind::Const(c) => {
                        // A constant reference
                    }
                    ValueKind::PrimOp(op) => {
                        let reads = fun.primop_reads(op);
                        match fun.primop_kind(op) {
                            PrimOpKind::ValueList => {

                            }
                            PrimOpKind::Tuple => {

                            }
                            PrimOpKind::ListCell => {

                            }
                            PrimOpKind::BinOp(binop) => {

                            }
                            other => unimplemented!("{:?}", other),
                        }
                    }
                }
            }).collect();
            // Build call
        }
        // i8*, module, function, arg
        OpKind::CaptureFunction => {
            // TODO: What is reads[0] supposed to be?
            let fun = reads[0];
            // Extract terms for module/fun/arity
            let m = reads[1];
            let f = reads[2];
            let a = reads[3];

        }
        // true: i8*, false: i8*, else: Option<i8*>, value
        // If 'else' is None, it is an unreachable block
        OpKind::IfBool => {}
        // symbol + per-intrinsic args
        OpKind::Intrinsic => {
        }
        OpKind::Match { _branches } => {
        }
        OpKind::Unreachable => {
            // Nothing more to do, so return early here
            builder.build_unreachable();
            return;
        }
        op => {
            panic!("unexpected operation in entry block of {}: {:?}", &function_name, op);
        }
    }
}

fn define_constant(constant: Const, kind: ConstKind, int_ty: IntType, ctx: &Context, module: &LLVMModule) {
    use inkwell::AddressSpace;
    use liblumen_alloc::erts::term::arch::{arch64, arch32};

    match kind {
        ConstKind::Atomic(AtomicTerm::Nil) => {
            if module.get_global("NIL").is_none() {
                let nil_value = match int_ty.get_bit_width() {
                    32 => arch32::FLAG_NIL,
                    64 => arch64::FLAG_NIL,
                    _ => unreachable!()
                };
                let const_val = int_ty.const_int(nil_value, false);
                let global = module.add_global(int_ty, Some(AddressSpace::Const), "NIL");
                global.set_initializer(const_val);
                global.set_alignment(8);
            }
            return;
        }
        // We define globals for every atom, with metadata that marks the global as an atom
        // Then later, prior to linking modules and during generation of the entry point module:
        //   - We gather all of the globals which are atoms
        //   - Generate a global in the entry module that contains an array of all the 
        //     statically known atoms
        //   - Generate a call to the runtime in the entry point which seeds the atom table 
        //     with the atoms in that array
        //   - Last but not least, replace all usages of each atom global with the calculated 
        //     atom id (derived from the index in the seed array)
        ConstKind::Atomic(AtomicTerm::Atom(ref a)) => {
            let i8_ty = ctx.i8_type();
            let value = a.value();
            let bytes = value
                .as_bytes()
                .map(|b| i8_ty.const_int(b, false))
                .collect::<Vec<_>>();
            let ty = i8_ty.array_type(bytes.len());
            let value = i8_ty.const_array(bytes);
            let name = format!("atom{}", value.as_u32());
            let global = module.add_global(ty, Some(AddressSpace::Const), name);
            global.set_initializer(value);
        }
        // All other atomics are handled uniformly
        ConstKind::Atomic(atomic) => {
            let name = format!("const{}", constant.as_u32());
            let (const_ty, const_val) = atomic_term_to_ty(atomic, int_ty, ctx);
            let global = module.add_global(const_ty, Some(AddressSpace::Const), name);
            global.set_initializer(const_val);
            global.set_alignment(MIN_ALIGN);
            return;
        }
    }
}

fn atomic_term_to_ty(term: AtomicTerm, int_ty: IntType, ctx: &Context) -> (dyn BasicType, dyn BasicValue) {
    use inkwell::types::*;
    use liblumen_alloc::erts::term::arch::{arch64, arch32};

    let is_arch64 = match int_ty.get_bit_width() {
        32 => false,
        64 => true,
        _ => unreachable!()
    };

    match term {
        AtomicTerm::Int(_) => atomic_int_to_ty(term, int_ty, is_arch64),
        AtomicTerm::BigInt(_) => atomic_int_to_ty(term, int_ty, is_arch64),
        // Floats are an array of two words (header + value) on arch64
        AtomicTerm::Float(ref f) if is_arch64 => {
            let header = int_ty.const_int(arch64::make_header(1, arch64::FLAG_FLOAT), false);
            let fval = int_ty.const_int(f.value().to_bits(), false);
            let value = int_ty.const_array(&[header, fval]);
            let ty = int_ty.array_type(2);
            (ty, value)
        }
        // Floats are an array of three words (header + two value words) on arch32
        AtomicTerm::Float(ref f) => {
            let header = int_ty.const_int(arch32::make_header(2, arch32::FLAG_FLOAT) as u64, false);
            let fval = int_ty.const_int(f.value().to_bits(), false);
            let value = int_ty.const_array(&[header, fval]);
            let ty = int_ty.array_type(2);
            (ty, value)
        }
        // Binary constants are stored as raw byte arrays
        AtomicTerm::Binary(ref b) => {
            let i8_ty = ctx.i8_type();
            let value = b.value();
            let bytes = value
                .as_bytes()
                .map(|b| i8_ty.const_int(b, false))
                .collect::<Vec<_>>();
            let ty = i8_ty.array_type(bytes.len());
            let value = i8_ty.const_array(bytes);
            (ty, value)
        }
        // These atomic values must always be handled by the caller, so these clauses should never be reachable
        AtomicTerm::Atom(ref a) => unreachable!(),
        AtomicTerm::Nil => unreachable!()
    }
}

#[inline]
fn atomic_int_to_ty(term: AtomicTerm, int_ty: IntType, is_arch64: bool) -> (dyn BasicType, dyn BasicValue) {
    match term {
        AtomicTerm::Int(i) if is_arch64 => {
            match Term::make_integer_for_arch64(i.value()) {
                Arch64Integer::Small(small) => (int_ty, int_ty.const_int(small, false)),
                Arch64Integer::Big(ref big) => make_bigint_for_arch64(big.value(), int_ty),
            }
        }
        AtomicTerm::Int(i) => {
            match Term::make_integer_for_arch32(i.value()) {
                Arch32Integer::Small(small) => (int_ty, int_ty.const_int(small as u64, false)),
                Arch32Integer::Big(ref big) => make_bigint_for_arch32(big.value(), int_ty),
            }
        }
        AtomicTerm::BigInt(ref big) if is_arch64 => make_bigint_for_arch64(big.value(), int_ty),
        AtomicTerm::BigInt(ref big) => make_bigint_for_arch32(big.value(), int_ty),
        _ => unreachable!()
    }
}

// BigInt is encoded as an arbitrary precision LLVM integer, where the first word is the term header
#[inline]
fn make_bigint_for_arch64(big: &BigInt, int_ty: IntType) -> (ArrayType, ArrayValue) {
    let (_sign, bytes) = big.to_bytes_le();
    let mut words = bytes_to_llvm_words(int_ty, bytes);
    let header = int_ty.const_int(Term::make_bigint_header_for_arch64(big), false);
    words.insert(0, header);
    let ty = int_ty.array_type(words.len());
    let value = int_ty.const_array(words.as_slice());
    (ty, value)
}

// BigInt is encoded as an arbitrary precision LLVM integer, where the first word is the term header
#[inline]
fn make_bigint_for_arch32(big: &BigInt, int_ty: IntType) -> (ArrayType, ArrayValue) {
    let (_sign, bytes) = big.to_bytes_le();
    let mut words = bytes_to_llvm_words(int_ty, bytes);
    let header = int_ty.const_int(Term::make_bigint_header_for_arch32(big) as u64, false);
    words.insert(0, header);
    let ty = int_ty.array_type(words.len());
    let value = int_ty.const_array(words.as_slice());
    (ty, value)
}

#[inline]
fn bytes_to_llvm_words(int_ty: IntType, mut bytes: Vec<u8>) -> Vec<IntValue> {
    let mut words = Vec::new();
    let mut byte_slice = bytes.as_slice();
    loop {
        let len = byte_slice.len();
        if len >= 8 {
            let (word_bytes, rest) = byte_slice.split_at(mem::size_of::<u64>());
            *byte_slice = rest;
            let i = int_ty.const_int(u64::from_le_bytes(word_bytes.try_into().unwrap()), false);
            words.push(i);
        } else {
            let missing = mem::size_of::<u64>() - len;
            let word_bytes = byte_slice.to_vec();
            for _ in 0..missing {
                word_bytes.push(0x0);
            }
            assert_eq!(word_bytes.len(), 8);
            let i = int_ty.const_int(u64::from_le_bytes(word_bytes.try_into().unwrap()), false);
            words.push(i);
            break;
        }
    }
    words
}

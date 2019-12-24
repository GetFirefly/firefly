mod ffi;
use self::ffi::*;

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::convert::AsRef;
use std::ffi::CString;

use anyhow::anyhow;

use log::debug;

use libeir_intern::{Ident, Symbol};
use libeir_ir as eir;
use libeir_ir::{
    AtomTerm, AtomicTerm, BigIntTerm, BinaryTerm, ConstKind, FloatTerm, IntTerm, NilTerm,
};
use libeir_ir::{BasicType, BinOp, CallKind, LogicOp, MapPutUpdate, MatchKind, OpKind};
use libeir_ir::{Block, Function, FunctionIdent, PrimOp, PrimOpKind, Value, ValueKind};
use libeir_lowerutils::{FunctionData, LowerData};

use liblumen_session::Options;

use crate::mlir::*;
use crate::Result;

extern "C" {
    pub type LocationImpl;
}
extern "C" {
    pub type ModuleBuilderImpl;
}
extern "C" {
    pub type FunctionOpImpl;
}
extern "C" {
    pub type BlockImpl;
}

pub type LocationRef = *mut LocationImpl;
pub type ModuleBuilderRef = *mut ModuleBuilderImpl;
pub type FunctionOpRef = *mut FunctionOpImpl;
pub type BlockRef = *mut BlockImpl;

/// Constructs an MLIR module from an EIR module, using the provided context and options
pub fn build(module: &eir::Module, context: &Context, options: &Options) -> Result<Module> {
    let module_name = module.name();

    debug!("building mlir module for {}", module_name);
    let builder = ModuleBuilder::new(context, module_name);

    for function_def in module.function_iter() {
        let function = function_def.function();
        let function_name = function.ident();
        debug!("running lowering analysis for {}", function_name);
        let analysis = libeir_lowerutils::analyze(function);
        debug!("building mlir function {}", function_name);
        builder.build_function(function, analysis)?;
    }

    debug!("finalizing {}", module_name);
    builder.finish()
}

pub struct ModuleBuilder<'ctx> {
    builder: ModuleBuilderRef,
    context: &'ctx Context,
    atoms: RefCell<HashSet<Symbol>>,
}
impl<'ctx> ModuleBuilder<'ctx> {
    /// Creates a new builder for the given module
    pub fn new(context: &'ctx Context, name: Ident) -> Self {
        let c_name = CString::new(name.to_string()).unwrap();
        let builder = unsafe { MLIRCreateModuleBuilder(context.as_ref(), c_name.as_ptr()) };
        Self {
            builder,
            context,
            atoms: RefCell::new(HashSet::new()),
        }
    }

    pub fn build_function(&self, function: &Function, analysis: LowerData) -> Result<()> {
        // Gather atoms in this function and add them to the atom table for this module
        for val in function.iter_constants() {
            if let ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(s))) =
                self.value_to_const_kind(function, *val)
            {
                self.atoms.borrow_mut().insert(*s);
            }
        }

        // Build root function
        let name = function.ident();
        let entry_block = function.block_entry();
        let entry_data = &analysis.functions[&entry_block];
        self.build_scoped_function(function, *name, entry_data, &analysis)?;

        // Build lifted functions/closures
        for (block, data) in analysis.functions.iter() {
            if entry_block == *block {
                continue;
            }
            let arity = function.block_args(data.entry).len();
            let ident = FunctionIdent {
                module: name.module.clone(),
                name: Ident::from_str(&format!("{}-fun-{}", name.name, arity)),
                arity,
            };
            self.build_scoped_function(function, ident, data, &analysis)?;
        }

        Ok(())
    }

    fn build_scoped_function(
        &self,
        function: &Function,
        name: FunctionIdent,
        data: &FunctionData,
        analysis: &LowerData,
    ) -> Result<()> {
        let entry_block = data.entry;
        let ret = data
            .ret
            .expect("expected function to have return continuation");
        let esc = data
            .thr
            .expect("expected function to have escape continuation");

        // 1. Gather argument types and result type
        let block_args = function.block_args(entry_block);
        let mut args = Vec::with_capacity(block_args.len());
        for arg in block_args.iter().copied() {
            if let ValueKind::Argument(_, _) = function.value_kind(arg) {
                let loc = function
                    .value_locations(arg)
                    .map(|locs| {
                        let span = locs[0];
                        (span.start().to_usize() as u32, span.end().to_usize() as u32)
                    })
                    .unwrap_or((0, 0));

                args.push(Arg {
                    ty: Type::Term,
                    span_start: loc.0,
                    span_end: loc.1,
                });
            }
        }
        let result_type = Type::Term;

        // 2. Create function op
        let func_op = self.create_function(name.to_string(), args, result_type)?;

        // 3. Create entry block which sets up stack frame, etc.
        let init_bb = self.append_basic_block(func_op, /* is_entry= */ true)?;
        self.position_at_end(init_bb);

        // TODO: Generate stack frame

        // 4. If this is a closure, extract the environment
        // A closure will have more than 1 live value, otherwise it is a regular function
        let live = analysis.live.live_at(entry_block);
        if live.size() > 0 {
            panic!("closure env unpacking is unimplemented!");
        }

        // Make the init block branch to the EIR entry block
        let entry_bb = self.append_basic_block(func_op, /* is_entry= */ false)?;
        self.position_at_end(init_bb);
        self.build_unconditional_branch(entry_bb);

        // 5. Translate each EIR block to an MLIR block
        let bbs: HashMap<_, _> = data
            .scope
            .iter()
            .map(|block| {
                (
                    *block,
                    self.append_basic_block(func_op, /* is_entry= */ false)
                        .unwrap(),
                )
            })
            .collect();

        // Lower entry block
        let entry_block_kind = function.block_kind(entry_block).unwrap();
        self.build_block(function, func_op, entry_block, entry_bb, entry_block_kind)?;

        // Lower successor blocks
        for (block, bb) in bbs.iter() {
            let block_kind = function.block_kind(*block).unwrap();
            self.build_block(function, func_op, *block, *bb, block_kind)?;
        }

        Ok(())
    }

    fn build_block(
        &self,
        function: &Function,
        func_op: FunctionOpRef,
        block: Block,
        bb: BlockRef,
        block_kind: &OpKind,
    ) -> Result<()> {
        // Lower primops
        // Lower operation
        let args = function.block_args(block);
        let reads = function.block_reads(block);

        match block_kind {
            // i8*, Term...
            OpKind::Call(call_kind) => {
                let fun = reads[0];
                // Build arguments
                let args = reads[1..]
                    .iter()
                    .map(|r| {
                        match function.value_kind(*r) {
                            ValueKind::Block(b) => {
                                // A closure env
                                todo!("closure env");
                            }
                            ValueKind::Argument(_, _) => {
                                // The value itself
                                todo!("argument");
                            }
                            ValueKind::Const(c) => {
                                // A constant reference
                                todo!("constant");
                            }
                            ValueKind::PrimOp(op) => {
                                let reads = function.primop_reads(op);
                                match function.primop_kind(op) {
                                    PrimOpKind::Tuple => {
                                        todo!("tuple");
                                    }
                                    PrimOpKind::ListCell => {
                                        todo!("list cell");
                                    }
                                    PrimOpKind::BinOp(binop) => {
                                        todo!("binop: {:#?}", binop);
                                    }
                                    other => unimplemented!("{:?}", other),
                                }
                            }
                        }
                    })
                    .collect::<Vec<_>>();
                // Build call
                todo!("build call");
            }
            // true: i8*, false: i8*, else: Option<i8*>, value
            // If 'else' is None, it is an unreachable block
            OpKind::IfBool => {}
            // symbol + per-intrinsic args
            OpKind::Intrinsic(name) => {
                todo!("intrinsic {}", name);
            }
            OpKind::Match { branches: _ } => {
                todo!("match");
            }
            OpKind::Unreachable => {
                // Nothing more to do, so return early here
                self.build_unreachable();
            }
            op => todo!("{:#?}", op),
        }

        Ok(())
    }

    fn append_basic_block(&self, func_op: FunctionOpRef, is_entry: bool) -> Result<BlockRef> {
        let result = if is_entry {
            unsafe { MLIRAppendEntryBlock(self.builder, func_op) }
        } else {
            unsafe { MLIRAppendBasicBlock(self.builder, func_op) }
        };
        if result.is_null() {
            return Err(anyhow!("failed to append block (is_entry={})", is_entry));
        }
        Ok(result)
    }

    fn position_at_end(&self, block: BlockRef) {
        unsafe { MLIRBlockPositionAtEnd(self.builder, block) }
    }

    fn build_unconditional_branch(&self, dest: BlockRef) {
        unsafe { MLIRBuildBr(self.builder, dest) }
    }

    fn build_unreachable(&self) {
        unsafe { MLIRBuildUnreachable(self.builder) }
    }

    fn create_function(
        &self,
        name: String,
        args: Vec<Arg>,
        result_type: Type,
    ) -> Result<FunctionOpRef> {
        let c_name = CString::new(name).unwrap();
        let result = unsafe {
            MLIRCreateFunction(
                self.builder,
                c_name.as_ptr(),
                args.as_ptr(),
                args.len() as libc::c_uint,
                result_type,
            )
        };
        if result.is_null() {
            return Err(anyhow!(
                "failed to create function {}",
                c_name.to_string_lossy()
            ));
        }
        Ok(result)
    }

    fn value_to_const_kind<'f>(&self, function: &'f Function, val: Value) -> &'f ConstKind {
        let constant = function.value_const(val).expect("expected constant value");
        function.const_kind(constant)
    }

    pub fn build_constant(&self, constant_kind: ConstKind) -> Result<()> {
        match constant_kind {
            ConstKind::Atomic(ref _atomic) => todo!("atomic constant"),
            kind => todo!("{:#?}", kind),
        }
    }

    /// Finalizes the builder, returning the built module, or an error
    pub fn finish(self) -> Result<Module> {
        let result = unsafe { MLIRFinalizeModuleBuilder(self.builder) };
        if result.is_null() {
            return Err(anyhow!(
                "unexpected error occurred when lowering EIR module"
            ));
        }
        Ok(Module::new(result))
    }
}

extern "C" {
    pub fn MLIRCreateModuleBuilder(
        context: ContextRef,
        name: *const libc::c_char,
    ) -> ModuleBuilderRef;

    pub fn MLIRFinalizeModuleBuilder(builder: ModuleBuilderRef) -> ModuleRef;

    pub fn MLIRCreateFunction(
        builder: ModuleBuilderRef,
        name: *const libc::c_char,
        argv: *const Arg,
        argc: libc::c_uint,
        result_type: Type,
    ) -> FunctionOpRef;

    pub fn MLIRAppendEntryBlock(builder: ModuleBuilderRef, fun: FunctionOpRef) -> BlockRef;

    pub fn MLIRAppendBasicBlock(builder: ModuleBuilderRef, fun: FunctionOpRef) -> BlockRef;

    pub fn MLIRBlockPositionAtEnd(builder: ModuleBuilderRef, block: BlockRef);

    pub fn MLIRBuildBr(builder: ModuleBuilderRef, dest: BlockRef);

    pub fn MLIRBuildUnreachable(builder: ModuleBuilderRef);

    pub fn MLIRCreateLocation(
        context: ContextRef,
        filename: *const libc::c_char,
        line: libc::c_uint,
        column: libc::c_uint,
    ) -> LocationRef;
}

#[no_mangle]
pub unsafe extern "C" fn EIRSpanToMLIRLocation(
    _start: libc::c_uint,
    _end: libc::c_uint,
) -> LocationRef {
    unimplemented!()
}

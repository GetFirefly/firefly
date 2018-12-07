#![allow(dead_code)]

mod enums;
mod memory_buffer;
mod target;

use std::sync::{Once, ONCE_INIT};

use llvm_sys::core::*;
use llvm_sys::prelude::*;
use llvm_sys::target::*;
use llvm_sys::*;

use codemap_diagnostic::{ColorConfig, Diagnostic, Emitter, Level};

use super::CodeGenError;

pub use self::enums::*;
use self::target::{Target, TargetMachine};

// Used to ensure LLVM is only initialized once
static ONCE: Once = ONCE_INIT;

/// Initialize LLVM internals
pub fn initialize() {
    ONCE.call_once(|| unsafe {
        error_handling::LLVMEnablePrettyStackTrace();

        LLVM_InitializeAllTargetInfos();
        LLVM_InitializeAllTargets();
        LLVM_InitializeAllTargetMCs();
        LLVM_InitializeAllAsmParsers();
        LLVM_InitializeAllAsmPrinters();

        let registry = LLVMGetGlobalPassRegistry();
        initialization::LLVMInitializeCore(registry);
        initialization::LLVMInitializeTarget(registry);
        initialization::LLVMInitializeCodeGen(registry);
        initialization::LLVMInitializeAnalysis(registry);
        initialization::LLVMInitializeTransformUtils(registry);
    });
}

/// The handler function for LLVM diagnostics
extern "C" fn diganostic_handler(info: LLVMDiagnosticInfoRef, ctx: *mut libc::c_void) {
    let ctx: &mut Context = unsafe { &mut *(ctx as *mut Context) };
    let severity = unsafe { LLVMGetDiagInfoSeverity(info) };
    let description = unsafe { LLVMGetDiagInfoDescription(info) };
    let d = Diagnostic {
        level: severity_to_diagnostic_level(severity),
        message: c_str_to_str!(description).to_string(),
        code: None,
        spans: Vec::new(),
    };
    ctx.emitter.emit(&vec![d]);
}

fn severity_to_diagnostic_level(severity: LLVMDiagnosticSeverity) -> Level {
    match severity {
        LLVMDiagnosticSeverity::LLVMDSError => Level::Error,
        LLVMDiagnosticSeverity::LLVMDSWarning => Level::Warning,
        LLVMDiagnosticSeverity::LLVMDSRemark => Level::Help,
        LLVMDiagnosticSeverity::LLVMDSNote => Level::Note,
    }
}

pub struct Context<'a> {
    ctx: LLVMContextRef,
    emitter: Emitter<'a>,
    target: Target,
    machine: TargetMachine,
    builder: Builder,
}
impl<'a> Context<'a> {
    pub fn new() -> Result<Context<'a>, CodeGenError> {
        let ctx = unsafe { LLVMContextCreate() };
        let emitter = Emitter::stderr(ColorConfig::Auto, None);
        // Create context
        let target = Target::default()?;
        let machine = TargetMachine::new(&target);
        let builder = Builder::new(unsafe { LLVMCreateBuilderInContext(ctx) });
        Ok(Context {
            ctx,
            emitter,
            target,
            machine,
            builder,
        })
    }

    pub fn set_diagnostic_handler(&mut self, handler: LLVMDiagnosticHandler) {
        let ptr: *mut libc::c_void = self as *mut _ as *mut libc::c_void;
        unsafe { LLVMContextSetDiagnosticHandler(self.ctx, handler, ptr) }
    }

    pub fn new_block(&self, fun: Function, name: &str) -> Block {
        let blk = unsafe { LLVMAppendBasicBlockInContext(self.ctx, fun.fun, c_str!(name)) };
        Block::new(blk)
    }

    pub fn new_block_before(&self, blk: Block, name: &str) -> Block {
        let blk = unsafe { LLVMInsertBasicBlockInContext(self.ctx, blk.blk, c_str!(name)) };
        //TODO
        Block::new(blk)
    }
}
impl<'a> std::convert::Into<LLVMContextRef> for Context<'a> {
    fn into(self) -> LLVMContextRef {
        self.ctx
    }
}

pub struct Builder {
    bldr: LLVMBuilderRef,
}
impl Builder {
    pub fn new(bldr: LLVMBuilderRef) -> Builder {
        Builder { bldr }
    }

    pub fn set_position(&self, block: Block, instruction: Value) {
        unsafe { LLVMPositionBuilder(self.bldr, block.blk, instruction.value_ref()) }
    }

    pub fn get_insert_block(&self) -> Block {
        Block::new(unsafe { LLVMGetInsertBlock(self.bldr) })
    }

    pub fn insert_block_before(&self, _before: Block, _blk: Block) {
        unimplemented!()
    }
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe { LLVMDisposeBuilder(self.bldr) };
    }
}

pub struct Function {
    fun: LLVMValueRef,
}
impl Function {
    pub fn new(fun: LLVMValueRef) -> Function {
        Function { fun }
    }

    pub fn append_block(&self, id: &str) -> Block {
        Block::new(unsafe { LLVMAppendBasicBlock(self.fun, c_str!(id)) })
    }

    pub fn num_blocks(&self) -> u32 {
        let n = unsafe { LLVMCountBasicBlocks(self.fun) };
        n as u32
    }

    pub fn entry_block(&self) -> Block {
        Block::new(unsafe { LLVMGetEntryBasicBlock(self.fun) })
    }

    pub fn first_block(&self) -> Block {
        Block::new(unsafe { LLVMGetFirstBasicBlock(self.fun) })
    }

    pub fn last_block(&self) -> Block {
        Block::new(unsafe { LLVMGetLastBasicBlock(self.fun) })
    }
}

pub struct Block {
    blk: LLVMBasicBlockRef,
}
impl Block {
    pub fn new(blk: LLVMBasicBlockRef) -> Block {
        Block { blk }
    }

    pub fn name(&self) -> &str {
        let ptr = unsafe { LLVMGetBasicBlockName(self.blk) };
        c_str_to_str!(ptr)
    }

    pub fn parent(&self) -> LLVMValueRef {
        unsafe { LLVMGetBasicBlockParent(self.blk) }
    }

    pub fn address(&self) -> Value {
        Value::BlockAddress(unsafe { LLVMBlockAddress(self.parent(), self.blk) })
    }

    pub fn new_before(&self, name: &str) -> Block {
        Block::new(unsafe { LLVMInsertBasicBlock(self.blk, c_str!(name)) })
    }

    pub fn next(&self) -> Option<Block> {
        let ptr = unsafe { LLVMGetNextBasicBlock(self.blk) };
        if ptr == std::ptr::null_mut() {
            None
        } else {
            Some(Block::new(ptr))
        }
    }

    pub fn previous(&self) -> Option<Block> {
        let ptr = unsafe { LLVMGetPreviousBasicBlock(self.blk) };
        if ptr == std::ptr::null_mut() {
            None
        } else {
            Some(Block::new(ptr))
        }
    }

    pub fn terminator(&self) -> Option<LLVMValueRef> {
        let ptr = unsafe { LLVMGetBasicBlockTerminator(self.blk) };
        if ptr == std::ptr::null_mut() {
            None
        } else {
            Some(ptr)
        }
    }

    pub fn delete(&self) {
        unsafe { LLVMDeleteBasicBlock(self.blk) };
    }
}

pub enum Value {
    Argument(LLVMValueRef),
    BasicBlock(LLVMValueRef),
    MemoryUse(LLVMValueRef),
    MemoryDef(LLVMValueRef),
    Phi(LLVMValueRef),
    Function(LLVMValueRef),
    GlobalAlias(LLVMValueRef),
    GlobalIFunc(LLVMValueRef),
    GlobalVariable(LLVMValueRef),
    BlockAddress(LLVMValueRef),
    ConstantExpr(LLVMValueRef),
    ConstantArray(LLVMValueRef),
    ConstantStruct(LLVMValueRef),
    ConstantVector(LLVMValueRef),
    Undef(LLVMValueRef),
    ConstantAggregateZero(LLVMValueRef),
    ConstantDataArray(LLVMValueRef),
    ConstantDataVector(LLVMValueRef),
    ConstantInt(LLVMValueRef),
    ConstantFunctionPtr(LLVMValueRef),
    ConstantNullPtr(LLVMValueRef),
    ConstantTokenNone(LLVMValueRef),
    MetadataValue(LLVMValueRef),
    InlineAsm(LLVMValueRef),
    Instruction(LLVMValueRef),
}
impl Value {
    pub fn value_ref(&self) -> LLVMValueRef {
        use self::Value::*;
        match *self {
            Argument(val) => val,
            BasicBlock(val) => val,
            MemoryUse(val) => val,
            MemoryDef(val) => val,
            Phi(val) => val,
            Function(val) => val,
            GlobalAlias(val) => val,
            GlobalIFunc(val) => val,
            GlobalVariable(val) => val,
            BlockAddress(val) => val,
            ConstantExpr(val) => val,
            ConstantArray(val) => val,
            ConstantStruct(val) => val,
            ConstantVector(val) => val,
            Undef(val) => val,
            ConstantAggregateZero(val) => val,
            ConstantDataArray(val) => val,
            ConstantDataVector(val) => val,
            ConstantInt(val) => val,
            ConstantFunctionPtr(val) => val,
            ConstantNullPtr(val) => val,
            ConstantTokenNone(val) => val,
            MetadataValue(val) => val,
            InlineAsm(val) => val,
            Instruction(val) => val,
        }
    }

    pub fn is_block_address(&self) -> bool {
        match *self {
            Value::BlockAddress(_) => true,
            _ => false,
        }
    }

    pub fn is_basic_block(&self) -> bool {
        match *self {
            Value::BasicBlock(_) => true,
            _ => false,
        }
    }
}
impl std::convert::From<Function> for Value {
    fn from(fun: Function) -> Value {
        Value::Function(fun.fun)
    }
}
impl std::convert::From<Block> for Value {
    fn from(blk: Block) -> Value {
        Value::BasicBlock(unsafe { LLVMBasicBlockAsValue(blk.blk) })
    }
}
impl std::convert::From<LLVMBasicBlockRef> for Value {
    fn from(blk: LLVMBasicBlockRef) -> Value {
        Value::BasicBlock(unsafe { LLVMBasicBlockAsValue(blk) })
    }
}

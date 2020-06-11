use std::cell::{Cell, RefCell};
use std::convert::Into;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::ptr;

use anyhow::anyhow;
use fxhash::FxHashMap;

use liblumen_session::{OptLevel, Options, Sanitizer};

use crate::sys as llvm_sys;
use crate::sys::prelude::LLVMBuilderRef;
use crate::sys::LLVMIntPredicate;

use crate::attributes::{Attribute, AttributePlace};
use crate::context::Context;
use crate::enums::{self, *};
use crate::funclet::Funclet;
use crate::module::Module;
use crate::passes::{PassBuilderOptLevel, PassManager};
use crate::target::{TargetData, TargetMachine};
use crate::{Block, Result, Type, Value};

/// Empty string, to be used where LLVM expects an instruction name, indicating
/// that the instruction is to be left unnamed (i.e. numbered, in textual IR).
const EMPTY_C_STR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") };
const UNNAMED: *const ::libc::c_char = EMPTY_C_STR.as_ptr();
const TARGET_CPU_STR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"target-cpu\0") };

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum ICmp {
    Eq,
    Neq,
    UGT,
    UGE,
    ULT,
    ULE,
    SGT,
    SGE,
    SLT,
    SLE,
}
impl Into<LLVMIntPredicate> for ICmp {
    fn into(self) -> LLVMIntPredicate {
        match self {
            Self::Eq => LLVMIntPredicate::LLVMIntEQ,
            Self::Neq => LLVMIntPredicate::LLVMIntNE,
            Self::UGT => LLVMIntPredicate::LLVMIntUGT,
            Self::UGE => LLVMIntPredicate::LLVMIntUGE,
            Self::ULT => LLVMIntPredicate::LLVMIntULT,
            Self::ULE => LLVMIntPredicate::LLVMIntULT,
            Self::SGT => LLVMIntPredicate::LLVMIntSGT,
            Self::SGE => LLVMIntPredicate::LLVMIntSGE,
            Self::SLT => LLVMIntPredicate::LLVMIntSLT,
            Self::SLE => LLVMIntPredicate::LLVMIntSLE,
        }
    }
}

pub struct ModuleBuilder<'ctx> {
    context: &'ctx Context,
    module: Module,
    target_data: TargetData,
    target_machine: &'ctx TargetMachine,
    builder: LLVMBuilderRef,
    intrinsics: RefCell<FxHashMap<&'static str, Value>>,
    local_gen_sym_counter: Cell<usize>,
    opt_level: PassBuilderOptLevel,
    sanitizer: Option<Sanitizer>,
    target_cpu: String,
    debug: bool,
    verify: bool,
}
impl<'ctx> ModuleBuilder<'ctx> {
    pub fn new(
        name: &str,
        options: &Options,
        context: &'ctx Context,
        target_machine: &'ctx TargetMachine,
    ) -> Result<Self> {
        use llvm_sys::core::LLVMCreateBuilderInContext;

        let target_data = target_machine.get_target_data();
        let module = Module::create(name, context, target_machine.as_ref())?;
        let builder = unsafe { LLVMCreateBuilderInContext(context.as_ref()) };

        let (speed, size) = enums::to_llvm_opt_settings(options.opt_level);
        let opt_level = PassBuilderOptLevel::from_codegen_opts(speed, size);

        Ok(Self {
            context,
            module,
            target_data,
            target_machine,
            builder,
            intrinsics: RefCell::new(Default::default()),
            local_gen_sym_counter: Cell::new(0),
            opt_level: opt_level,
            sanitizer: options.debugging_opts.sanitizer.clone(),
            target_cpu: crate::target::target_cpu(options).to_owned(),
            debug: options.debug_assertions,
            verify: options.debugging_opts.verify_llvm_ir,
        })
    }

    pub fn verify(&self) -> anyhow::Result<()> {
        use llvm_sys::analysis::{LLVMVerifierFailureAction, LLVMVerifyModule};

        let mut error_raw = MaybeUninit::<*mut libc::c_char>::uninit();
        let failed = unsafe {
            LLVMVerifyModule(
                self.module.as_ref(),
                LLVMVerifierFailureAction::LLVMPrintMessageAction,
                error_raw.as_mut_ptr(),
            ) == 1
        };

        if failed {
            let module_name = self.module.get_module_id();
            let error = unsafe { CStr::from_ptr(error_raw.assume_init()) };
            let detail = error.to_string_lossy();
            Err(anyhow!(format!(
                "failed to verify {}: {}",
                module_name, detail
            )))
        } else {
            Ok(())
        }
    }

    pub fn finish(self) -> anyhow::Result<Module> {
        let mut pass_manager = PassManager::new();
        pass_manager.verify(self.verify);
        pass_manager.debug(self.debug);
        pass_manager.optimize(self.opt_level);

        if let Some(sanitizer) = self.sanitizer {
            match sanitizer {
                Sanitizer::Memory => pass_manager.sanitize_memory(/* track_origins */ 0),
                Sanitizer::Thread => pass_manager.sanitize_thread(),
                Sanitizer::Address => pass_manager.sanitize_address(),
                _ => (),
            }
        }

        let mut module = self.module;

        pass_manager.run(&mut module, &self.target_machine)?;

        Ok(module)
    }

    #[inline]
    pub fn type_of(&self, value: Value) -> Type {
        unsafe { llvm_sys::core::LLVMTypeOf(value) }
    }

    #[inline]
    pub fn get_void_type(&self) -> Type {
        use llvm_sys::core::LLVMVoidTypeInContext;

        unsafe { LLVMVoidTypeInContext(self.context.as_ref()) }
    }

    pub fn get_i1_type(&self) -> Type {
        use llvm_sys::core::LLVMInt1TypeInContext;

        unsafe { LLVMInt1TypeInContext(self.context.as_ref()) }
    }

    pub fn get_i8_type(&self) -> Type {
        use llvm_sys::core::LLVMInt8TypeInContext;

        unsafe { LLVMInt8TypeInContext(self.context.as_ref()) }
    }

    pub fn get_i16_type(&self) -> Type {
        use llvm_sys::core::LLVMInt16TypeInContext;

        unsafe { LLVMInt16TypeInContext(self.context.as_ref()) }
    }

    pub fn get_i32_type(&self) -> Type {
        use llvm_sys::core::LLVMInt32TypeInContext;

        unsafe { LLVMInt32TypeInContext(self.context.as_ref()) }
    }

    pub fn get_i64_type(&self) -> Type {
        use llvm_sys::core::LLVMInt64TypeInContext;

        unsafe { LLVMInt64TypeInContext(self.context.as_ref()) }
    }

    pub fn get_i128_type(&self) -> Type {
        use llvm_sys::core::LLVMInt128TypeInContext;

        unsafe { LLVMInt128TypeInContext(self.context.as_ref()) }
    }

    pub fn get_usize_type(&self) -> Type {
        use llvm_sys::target::LLVMIntPtrTypeInContext;

        unsafe { LLVMIntPtrTypeInContext(self.context.as_ref(), self.target_data.as_ref()) }
    }

    #[inline]
    pub fn get_term_type(&self) -> Type {
        self.get_usize_type()
    }

    pub fn get_integer_type(&self, width: usize) -> Type {
        use llvm_sys::core::LLVMIntTypeInContext;

        unsafe { LLVMIntTypeInContext(self.context.as_ref(), width as libc::c_uint) }
    }

    pub fn get_f32_type(&self) -> Type {
        use llvm_sys::core::LLVMFloatTypeInContext;

        unsafe { LLVMFloatTypeInContext(self.context.as_ref()) }
    }

    pub fn get_f64_type(&self) -> Type {
        use llvm_sys::core::LLVMDoubleTypeInContext;

        unsafe { LLVMDoubleTypeInContext(self.context.as_ref()) }
    }

    pub fn get_metadata_type(&self) -> Type {
        use llvm_sys::core::LLVMMetadataTypeInContext;

        unsafe { LLVMMetadataTypeInContext(self.context.as_ref()) }
    }

    pub fn get_token_type(&self) -> Type {
        use llvm_sys::core::LLVMTokenTypeInContext;

        unsafe { LLVMTokenTypeInContext(self.context.as_ref()) }
    }

    pub fn get_vector_type(&self, element_ty: Type, size: usize) -> Type {
        use llvm_sys::core::LLVMVectorType;

        unsafe { LLVMVectorType(element_ty, size as libc::c_uint) }
    }

    pub fn get_array_type(&self, size: usize, ty: Type) -> Type {
        use llvm_sys::core::LLVMArrayType;

        unsafe { LLVMArrayType(ty, size as libc::c_uint) }
    }

    pub fn get_struct_type(&self, name: Option<&str>, field_types: &[Type]) -> Type {
        use llvm_sys::core::LLVMStructCreateNamed;
        use llvm_sys::core::LLVMStructSetBody;
        use llvm_sys::core::LLVMStructTypeInContext;

        if let Some(name) = name {
            let cstr = CString::new(name).unwrap();
            unsafe {
                let ty = LLVMStructCreateNamed(self.context.as_ref(), cstr.as_ptr());
                LLVMStructSetBody(
                    ty,
                    field_types.as_ptr() as *mut _,
                    field_types.len() as libc::c_uint,
                    /* packed= */ false as libc::c_int,
                );
                ty
            }
        } else {
            unsafe {
                LLVMStructTypeInContext(
                    self.context.as_ref(),
                    field_types.as_ptr() as *mut _,
                    field_types.len() as libc::c_uint,
                    /* packed= */ false as libc::c_int,
                )
            }
        }
    }

    pub fn get_function_type(&self, ret: Type, params: &[Type], variadic: bool) -> Type {
        use llvm_sys::core::LLVMFunctionType;

        let params_ptr = params.as_ptr() as *mut _;
        unsafe {
            LLVMFunctionType(
                ret,
                params_ptr,
                params.len() as libc::c_uint,
                variadic as libc::c_int,
            )
        }
    }

    pub fn get_erlang_function_type(&self, arity: usize) -> Type {
        let term_type = self.get_term_type();
        let mut params = Vec::with_capacity(arity);
        for _ in 0..arity {
            params.push(term_type);
        }
        self.get_function_type(term_type, params.as_slice(), /* variadic */ false)
    }

    pub fn get_opaque_function_type(&self) -> Type {
        let void_type = self.get_void_type();
        self.get_function_type(void_type, &[], /* variadic */ false)
    }

    pub fn get_pointer_type(&self, ty: Type) -> Type {
        use llvm_sys::core::LLVMPointerType;

        unsafe {
            LLVMPointerType(ty, /* address_space= */ 0 as libc::c_uint)
        }
    }

    pub fn build_is_null(&self, value: Value) -> Value {
        use llvm_sys::core::LLVMBuildIsNull;

        unsafe { LLVMBuildIsNull(self.builder, value, UNNAMED) }
    }

    pub fn build_bitcast(&self, value: Value, ty: Type) -> Value {
        use llvm_sys::core::LLVMBuildBitCast;

        unsafe { LLVMBuildBitCast(self.builder, value, ty, UNNAMED) }
    }

    pub fn build_inttoptr(&self, value: Value, ty: Type) -> Value {
        use llvm_sys::core::LLVMBuildIntToPtr;

        unsafe { LLVMBuildIntToPtr(self.builder, value, ty, UNNAMED) }
    }

    pub fn build_ptrtoint(&self, value: Value, ty: Type) -> Value {
        use llvm_sys::core::LLVMBuildPtrToInt;

        unsafe { LLVMBuildPtrToInt(self.builder, value, ty, UNNAMED) }
    }

    pub fn build_and(&self, lhs: Value, rhs: Value) -> Value {
        use llvm_sys::core::LLVMBuildAnd;

        unsafe { LLVMBuildAnd(self.builder, lhs, rhs, UNNAMED) }
    }

    pub fn build_or(&self, lhs: Value, rhs: Value) -> Value {
        use llvm_sys::core::LLVMBuildOr;

        unsafe { LLVMBuildOr(self.builder, lhs, rhs, UNNAMED) }
    }

    pub fn build_xor(&self, lhs: Value, rhs: Value) -> Value {
        use llvm_sys::core::LLVMBuildXor;

        unsafe { LLVMBuildXor(self.builder, lhs, rhs, UNNAMED) }
    }

    pub fn build_not(&self, value: Value) -> Value {
        use llvm_sys::core::LLVMBuildNot;

        unsafe { LLVMBuildNot(self.builder, value, UNNAMED) }
    }

    pub fn build_undef(&self, ty: Type) -> Value {
        use llvm_sys::core::LLVMGetUndef;

        unsafe { LLVMGetUndef(ty) }
    }

    pub fn build_constant_null(&self, ty: Type) -> Value {
        use llvm_sys::core::LLVMConstNull;

        unsafe { LLVMConstNull(ty) }
    }

    pub fn build_constant_int(&self, ty: Type, value: i64) -> Value {
        use llvm_sys::core::LLVMConstInt;

        unsafe {
            LLVMConstInt(
                ty,
                value as libc::c_ulonglong,
                /* sign_extend= */ true as libc::c_int,
            )
        }
    }

    pub fn build_constant_uint(&self, ty: Type, value: u64) -> Value {
        use llvm_sys::core::LLVMConstInt;

        unsafe {
            LLVMConstInt(
                ty,
                value as libc::c_ulonglong,
                /* sign_extend= */ false as libc::c_int,
            )
        }
    }

    pub fn build_constant_array(&self, ty: Type, values: &[Value]) -> Value {
        use llvm_sys::core::LLVMConstArray;

        let len = values.len() as libc::c_uint;
        unsafe { LLVMConstArray(ty, values.as_ptr() as *mut _, len) }
    }

    pub fn build_constant_vector(&self, elements: &[Value]) -> Value {
        use llvm_sys::core::LLVMConstVector;

        let len = elements.len() as libc::c_uint;
        unsafe { LLVMConstVector(elements.as_ptr() as *mut _, len) }
    }

    pub fn build_constant_bytes(&self, bytes: &[u8]) -> Value {
        use llvm_sys::core::LLVMConstStringInContext;

        unsafe {
            let ptr = bytes.as_ptr() as *const libc::c_char;
            LLVMConstStringInContext(
                self.context.as_ref(),
                ptr,
                bytes.len() as libc::c_uint,
                true as i32,
            )
        }
    }

    pub fn build_named_constant_string(&self, name: &str, s: &str, null_terminated: bool) -> Value {
        use llvm_sys::core::{LLVMConstStringInContext, LLVMSetGlobalConstant};

        unsafe {
            let sc = LLVMConstStringInContext(
                self.context.as_ref(),
                s.as_ptr() as *const libc::c_char,
                s.len() as libc::c_uint,
                !null_terminated as i32,
            );
            let g = self
                .define_global(name, self.type_of(sc))
                .unwrap_or_else(|| {
                    panic!("symbol `{}` is already defined", name);
                });
            self.set_initializer(g, sc);
            self.set_linkage(g, Linkage::Internal);

            g
        }
    }

    pub fn build_constant_string(&self, s: &str, null_terminated: bool) -> Value {
        let sym = self.generate_local_symbol_name("str");

        self.build_named_constant_string(&sym[..], s, null_terminated)
    }

    pub fn build_constant_struct(&self, ty: Type, fields: &[Value]) -> Value {
        use llvm_sys::core::LLVMConstNamedStruct;
        use llvm_sys::core::LLVMConstStructInContext;
        use llvm_sys::core::LLVMGetStructName;

        let name = unsafe { LLVMGetStructName(ty) };
        if name.is_null() {
            self.build_constant_unnamed_struct(fields)
        } else {
            unsafe {
                LLVMConstNamedStruct(ty, fields.as_ptr() as *mut _, fields.len() as libc::c_uint)
            }
        }
    }

    pub fn build_constant_unnamed_struct(&self, elements: &[Value]) -> Value {
        use llvm_sys::core::LLVMConstStructInContext;

        unsafe {
            LLVMConstStructInContext(
                self.context.as_ref(),
                elements.as_ptr() as *mut _,
                elements.len() as libc::c_uint,
                /* packed= */ false as libc::c_int,
            )
        }
    }

    pub fn build_constant_get_element(&self, agg: Value, index: u64) -> Value {
        use llvm_sys::core::LLVMConstExtractValue;

        unsafe {
            let indices = &[index as libc::c_uint];
            LLVMConstExtractValue(
                agg,
                indices.as_ptr() as *mut _,
                indices.len() as libc::c_uint,
            )
        }
    }

    pub fn declare_function(&self, name: &str, ty: Type) -> Value {
        use llvm_sys::core::LLVMLumenGetOrInsertFunction;

        let fun = unsafe {
            LLVMLumenGetOrInsertFunction(self.module.as_ref(), name.as_ptr().cast(), name.len(), ty)
        };

        self.apply_default_function_attributes(fun);

        fun
    }

    #[inline]
    pub fn build_function(&self, name: &str, ty: Type) -> Value {
        self.build_function_with_attrs(name, ty, Linkage::Internal, &[])
    }

    #[inline]
    pub fn build_external_function(&self, name: &str, ty: Type) -> Value {
        self.build_function_with_attrs(name, ty, Linkage::External, &[])
    }

    pub fn build_function_with_attrs(
        &self,
        name: &str,
        ty: Type,
        linkage: Linkage,
        attrs: &[Attribute],
    ) -> Value {
        let fun = self.declare_function(name, ty);

        for attr in attrs.iter().copied() {
            self.set_function_attr(fun, attr);
        }

        self.set_linkage(fun, linkage);

        fun
    }

    pub fn set_personality(&self, fun: Value, personality_fun: Value) {
        use llvm_sys::core::LLVMSetPersonalityFn;

        unsafe {
            LLVMSetPersonalityFn(fun, personality_fun);
        }
    }

    pub fn set_function_attr(&self, fun: Value, attr: Attribute) {
        use crate::attributes::LLVMLumenAddFunctionAttribute;

        unsafe {
            LLVMLumenAddFunctionAttribute(fun, AttributePlace::Function.as_uint(), attr);
        }
    }

    pub fn set_function_attr_string_value(
        &self,
        fun: Value,
        idx: AttributePlace,
        attr: &CStr,
        value: &CStr,
    ) {
        use crate::attributes::LLVMLumenAddFunctionAttrStringValue;

        unsafe {
            LLVMLumenAddFunctionAttrStringValue(fun, idx.as_uint(), attr.as_ptr(), value.as_ptr());
        }
    }

    pub fn remove_function_attr(&self, fun: Value, attr: Attribute) {
        use crate::attributes::LLVMLumenRemoveFunctionAttributes;

        unsafe {
            LLVMLumenRemoveFunctionAttributes(fun, AttributePlace::Function.as_uint(), attr);
        }
    }

    pub fn set_callsite_attr(&self, call: Value, attr: Attribute, idx: AttributePlace) {
        use crate::attributes::LLVMLumenAddCallSiteAttribute;

        unsafe {
            LLVMLumenAddCallSiteAttribute(call, idx.as_uint(), attr);
        }
    }

    pub fn get_function_params(&self, fun: Value) -> Vec<Value> {
        use llvm_sys::core::{LLVMCountParams, LLVMGetParams};
        let paramc = unsafe { LLVMCountParams(fun) as usize };
        let mut params = Vec::with_capacity(paramc);
        unsafe {
            LLVMGetParams(fun, params.as_mut_ptr());
            params.set_len(paramc);
        };
        params
    }

    pub fn get_function_param(&self, fun: Value, index: usize) -> Value {
        use llvm_sys::core::LLVMGetParam;

        unsafe { LLVMGetParam(fun, index as libc::c_uint) }
    }

    #[inline]
    pub fn build_entry_block(&self, fun: Value) -> Block {
        self.build_named_block(fun, "entry")
    }

    pub fn build_block(&self, fun: Value) -> Block {
        use llvm_sys::core::LLVMAppendBasicBlockInContext;

        unsafe { LLVMAppendBasicBlockInContext(self.context.as_ref(), fun, UNNAMED) }
    }

    pub fn build_named_block(&self, fun: Value, name: &str) -> Block {
        use llvm_sys::core::LLVMAppendBasicBlockInContext;

        let name = CString::new(name).unwrap();
        unsafe { LLVMAppendBasicBlockInContext(self.context.as_ref(), fun, name.as_ptr()) }
    }

    pub fn position_at_end(&self, block: Block) {
        use llvm_sys::core::LLVMPositionBuilderAtEnd;

        unsafe {
            LLVMPositionBuilderAtEnd(self.builder, block);
        }
    }

    pub fn build_alloca(&self, ty: Type) -> Value {
        use llvm_sys::core::LLVMBuildAlloca;

        unsafe { LLVMBuildAlloca(self.builder, ty, UNNAMED) }
    }

    pub fn build_phi(&self, ty: Type, incoming: &[(Value, Block)]) -> Value {
        use llvm_sys::core::{LLVMAddIncoming, LLVMBuildPhi};

        let phi = unsafe { LLVMBuildPhi(self.builder, ty, UNNAMED) };

        let num_incoming = incoming.len() as libc::c_uint;
        let values = incoming.iter().map(|(v, _)| v).copied().collect::<Vec<_>>();
        let blocks = incoming.iter().map(|(_, b)| b).copied().collect::<Vec<_>>();

        unsafe {
            LLVMAddIncoming(
                phi,
                values.as_ptr() as *mut _,
                blocks.as_ptr() as *mut _,
                num_incoming,
            );
        }

        phi
    }

    pub fn build_br(&self, dest: Block) -> Value {
        use llvm_sys::core::LLVMBuildBr;

        unsafe { LLVMBuildBr(self.builder, dest) }
    }

    pub fn build_condbr(&self, cond: Value, then_dest: Block, else_dest: Block) -> Value {
        use llvm_sys::core::LLVMBuildCondBr;

        unsafe { LLVMBuildCondBr(self.builder, cond, then_dest, else_dest) }
    }

    pub fn build_switch(
        &self,
        value: Value,
        clauses: &[(Value, Block)],
        else_dest: Block,
    ) -> Value {
        use llvm_sys::core::{LLVMAddCase, LLVMBuildSwitch};

        let s = unsafe {
            LLVMBuildSwitch(
                self.builder,
                value,
                else_dest,
                clauses.len() as libc::c_uint,
            )
        };

        for (clause_value, clause_dest) in clauses.iter() {
            unsafe {
                LLVMAddCase(s, *clause_value, *clause_dest);
            }
        }

        s
    }

    pub fn build_icmp(&self, lhs: Value, rhs: Value, predicate: ICmp) -> Value {
        use llvm_sys::core::LLVMBuildICmp;

        unsafe { LLVMBuildICmp(self.builder, predicate.into(), lhs, rhs, UNNAMED) }
    }

    pub fn build_call(&self, fun: Value, args: &[Value], funclet: Option<&Funclet>) -> Value {
        use llvm_sys::core::LLVMLumenBuildCall;

        let argv = args.as_ptr() as *mut _;
        let argc = args.len() as libc::c_uint;
        let funclet = funclet
            .map(|f| f.bundle().as_ref() as *const _)
            .unwrap_or(ptr::null());

        unsafe { LLVMLumenBuildCall(self.builder, fun, argv, argc, funclet, UNNAMED) }
    }

    pub fn set_is_tail(&self, call: Value, is_tail: bool) {
        use llvm_sys::core::LLVMSetTailCall;
        use llvm_sys::prelude::LLVMBool;
        unsafe { LLVMSetTailCall(call, is_tail as LLVMBool) }
    }

    pub fn build_return(&self, ret: Value) -> Value {
        use llvm_sys::core::LLVMBuildRet;

        unsafe { LLVMBuildRet(self.builder, ret) }
    }

    pub fn build_unreachable(&self) -> Value {
        use llvm_sys::core::LLVMBuildUnreachable;

        unsafe { LLVMBuildUnreachable(self.builder) }
    }

    pub fn build_invoke(
        &self,
        fun: Value,
        args: &[Value],
        normal: Block,
        unwind: Block,
        funclet: Option<&Funclet>,
    ) -> Value {
        use llvm_sys::core::LLVMLumenBuildInvoke;

        let argv = args.as_ptr() as *mut _;
        let argc = args.len() as libc::c_uint;
        let funclet = funclet
            .map(|f| f.bundle().as_ref() as *const _)
            .unwrap_or(ptr::null());

        unsafe {
            LLVMLumenBuildInvoke(
                self.builder,
                fun,
                argv,
                argc,
                normal,
                unwind,
                funclet,
                UNNAMED,
            )
        }
    }

    pub fn build_resume(&self, exception: Value) -> Value {
        use llvm_sys::core::LLVMBuildResume;

        unsafe { LLVMBuildResume(self.builder, exception) }
    }

    pub fn build_landingpad(&self, ty: Type, personality_fun: Value, clauses: &[Value]) -> Value {
        use llvm_sys::core::{LLVMAddClause, LLVMBuildLandingPad};

        let num_clauses = clauses.len() as libc::c_uint;
        let pad =
            unsafe { LLVMBuildLandingPad(self.builder, ty, personality_fun, num_clauses, UNNAMED) };

        for clause in clauses.iter().copied() {
            unsafe {
                LLVMAddClause(pad, clause);
            }
        }

        pad
    }

    pub fn build_catchpad(&self, parent: Option<Value>, args: &[Value]) -> Funclet {
        use llvm_sys::core::LLVMBuildCatchPad;

        let argv = args.as_ptr() as *mut _;
        let argc = args.len() as libc::c_uint;
        let parent = match parent {
            None => self.build_constant_null(self.get_token_type()),
            Some(p) => p,
        };
        let pad = unsafe { LLVMBuildCatchPad(self.builder, parent, argv, argc, UNNAMED) };

        Funclet::new(pad)
    }

    pub fn build_cleanuppad(&self, parent: Option<Value>, args: &[Value]) -> Funclet {
        use llvm_sys::core::LLVMBuildCleanupPad;

        let argv = args.as_ptr() as *mut _;
        let argc = args.len() as libc::c_uint;
        let pad = unsafe {
            LLVMBuildCleanupPad(
                self.builder,
                parent.unwrap_or(ptr::null_mut() as _),
                argv,
                argc,
                UNNAMED,
            )
        };

        Funclet::new(pad)
    }

    pub fn build_catchret(&self, funclet: &Funclet, dest: Option<Block>) -> Value {
        use llvm_sys::core::LLVMBuildCatchRet;

        let dest = dest.unwrap_or(ptr::null_mut());
        unsafe { LLVMBuildCatchRet(self.builder, funclet.pad(), dest) }
    }

    pub fn build_cleanupret(&self, funclet: &Funclet, dest: Option<Block>) -> Value {
        use llvm_sys::core::LLVMBuildCleanupRet;

        let dest = dest.unwrap_or(ptr::null_mut());
        unsafe { LLVMBuildCleanupRet(self.builder, funclet.pad(), dest) }
    }

    pub fn build_catchswitch(
        &self,
        parent: Option<Value>,
        unwind: Option<Block>,
        handlers: &[Block],
    ) -> Value {
        use llvm_sys::core::{LLVMAddHandler, LLVMBuildCatchSwitch};

        let parent = match parent {
            None => self.build_constant_null(self.get_token_type()),
            Some(p) => p,
        };
        let unwind = unwind.unwrap_or(ptr::null_mut());
        let num_handlers = handlers.len() as libc::c_uint;
        let cs =
            unsafe { LLVMBuildCatchSwitch(self.builder, parent, unwind, num_handlers, UNNAMED) };

        for handler in handlers.iter().copied() {
            unsafe {
                LLVMAddHandler(cs, handler);
            }
        }

        cs
    }

    pub fn build_extractvalue(&self, agg: Value, index: usize) -> Value {
        use llvm_sys::core::LLVMBuildExtractValue;

        unsafe { LLVMBuildExtractValue(self.builder, agg, index as libc::c_uint, UNNAMED) }
    }

    pub fn build_insertvalue(&self, agg: Value, element: Value, index: usize) -> Value {
        use llvm_sys::core::LLVMBuildInsertValue;

        unsafe { LLVMBuildInsertValue(self.builder, agg, element, index as libc::c_uint, UNNAMED) }
    }

    pub fn build_load(&self, ty: Type, ptr: Value) -> Value {
        use llvm_sys::core::LLVMBuildLoad2;

        unsafe { LLVMBuildLoad2(self.builder, ty, ptr, UNNAMED) }
    }

    pub fn build_store(&self, ptr: Value, value: Value) -> Value {
        use llvm_sys::core::LLVMBuildStore;
        unsafe { LLVMBuildStore(self.builder, ptr, value) }
    }

    pub fn build_struct_gep(&self, ptr: Value, index: usize) -> Value {
        use llvm_sys::core::LLVMBuildStructGEP;

        unsafe { LLVMBuildStructGEP(self.builder, ptr, index as libc::c_uint, UNNAMED) }
    }

    pub fn build_inbounds_gep(&self, ty: Type, ptr: Value, indices: &[usize]) -> Value {
        use llvm_sys::core::LLVMBuildInBoundsGEP2;

        let i32_type = self.get_i32_type();
        let indices_values = indices
            .iter()
            .map(|i| self.build_constant_uint(i32_type, *i as u64))
            .collect::<Vec<_>>();
        let num_indices = indices_values.len() as libc::c_uint;
        unsafe {
            LLVMBuildInBoundsGEP2(
                self.builder,
                ty,
                ptr,
                indices_values.as_ptr() as *mut _,
                num_indices,
                UNNAMED,
            )
        }
    }

    pub fn build_constant(&self, ty: Type, name: &str, initializer: Option<Value>) -> Value {
        use llvm_sys::core::LLVMSetGlobalConstant;

        let global = self.build_global(ty, name, initializer);
        unsafe {
            LLVMSetGlobalConstant(global, true as libc::c_int);
        }
        global
    }

    pub fn build_global(&self, ty: Type, name: &str, initializer: Option<Value>) -> Value {
        use llvm_sys::core::LLVMAddGlobal;

        let cstr = CString::new(name).unwrap();
        let global = unsafe { LLVMAddGlobal(self.module.as_ref(), ty, cstr.as_ptr()) };
        if let Some(init) = initializer {
            self.set_initializer(global, init);
        }
        global
    }

    pub fn set_initializer(&self, global: Value, constant: Value) {
        use llvm_sys::core::LLVMSetInitializer;

        unsafe {
            LLVMSetInitializer(global, constant);
        }
    }

    pub fn set_linkage(&self, value: Value, linkage: Linkage) {
        use llvm_sys::core::LLVMSetLinkage;

        unsafe {
            LLVMSetLinkage(value, linkage.into());
        }
    }

    pub fn set_thread_local_mode(&self, global: Value, tls: ThreadLocalMode) {
        use llvm_sys::core::LLVMSetThreadLocalMode;

        unsafe {
            LLVMSetThreadLocalMode(global, tls.into());
        }
    }

    pub fn set_alignment(&self, value: Value, alignment: usize) {
        use llvm_sys::core::LLVMSetAlignment;

        unsafe {
            LLVMSetAlignment(value, alignment as libc::c_uint);
        }
    }

    pub fn build_pointer_cast(&self, value: Value, ty: Type) -> Value {
        use llvm_sys::core::LLVMConstPointerCast;

        unsafe { LLVMConstPointerCast(value, ty) }
    }

    pub fn build_const_inbounds_gep(&self, value: Value, indices: &[usize]) -> Value {
        use llvm_sys::core::LLVMConstInBoundsGEP;

        let i32_type = self.get_i32_type();
        let indices_values = indices
            .iter()
            .map(|i| self.build_constant_uint(i32_type, *i as u64))
            .collect::<Vec<_>>();
        let num_indices = indices_values.len() as libc::c_uint;
        unsafe { LLVMConstInBoundsGEP(value, indices_values.as_ptr() as *mut _, num_indices) }
    }

    pub fn declare_global(&self, name: &str, ty: Type) -> Value {
        use llvm_sys::core::LLVMLumenGetOrInsertGlobal;

        unsafe {
            LLVMLumenGetOrInsertGlobal(self.module.as_ref(), name.as_ptr().cast(), name.len(), ty)
        }
    }

    pub fn define_global(&self, name: &str, ty: Type) -> Option<Value> {
        if self.get_defined_value(name).is_some() {
            None
        } else {
            Some(self.declare_global(name, ty))
        }
    }

    pub fn get_declared_value(&self, name: &str) -> Option<Value> {
        use llvm_sys::core::LLVMGetNamedGlobal;

        let name = CString::new(name).unwrap();
        let g = unsafe { LLVMGetNamedGlobal(self.module.as_ref(), name.as_ptr()) };
        if g.is_null() {
            None
        } else {
            Some(g)
        }
    }

    pub fn get_defined_value(&self, name: &str) -> Option<Value> {
        use llvm_sys::core::LLVMIsDeclaration;

        self.get_declared_value(name).and_then(|val| {
            let declaration = unsafe { LLVMIsDeclaration(val) != 0 };
            if !declaration {
                Some(val)
            } else {
                None
            }
        })
    }

    /// Generates a new symbol name with the given prefix. This symbol name must
    /// only be used for definitions with `internal` or `private` linkage.
    pub fn generate_local_symbol_name(&self, prefix: &str) -> String {
        let idx = self.local_gen_sym_counter.get();
        self.local_gen_sym_counter.set(idx + 1);
        // Include a '.' character, so there can be no accidental conflicts with
        // user defined names
        let mut name = String::with_capacity(prefix.len() + 6);
        name.push_str(prefix);
        name.push_str(".");
        push_base_n_str(idx as u128, BASE_N_ALPHANUMERIC_ONLY, &mut name);
        name
    }

    // Externally visible symbols that might appear in multiple codegen units need to appear in
    // their own comdat section so that the duplicates can be discarded at link time. This can for
    // example happen for generics when using multiple codegen units. This function simply uses the
    // value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
    // function.
    // For more details on COMDAT sections see e.g., http://www.airs.com/blog/archives/52
    pub fn set_unique_comdat(&self, val: Value) {
        use llvm_sys::comdat::LLVMLumenSetComdat;

        unsafe {
            let name = self.get_value_name(val);
            LLVMLumenSetComdat(self.module.as_ref(), val, name.as_ptr().cast(), name.len());
        }
    }

    pub fn get_value_name(&self, value: Value) -> &[u8] {
        use llvm_sys::core::LLVMGetValueName2;

        unsafe {
            let mut len = 0;
            let data = LLVMGetValueName2(value, &mut len);
            std::slice::from_raw_parts(data.cast(), len)
        }
    }

    pub fn unset_comdat(&self, val: Value) {
        use llvm_sys::comdat::LLVMLumenUnsetComdat;

        unsafe {
            LLVMLumenUnsetComdat(val);
        }
    }

    fn apply_default_function_attributes(&self, fun: Value) {
        self.apply_optimization_attributes(fun);
        self.apply_sanitizers(fun);
        // Always annotate functions with the target-cpu they are compiled for.
        // Without this, ThinLTO won't inline Rust functions into Clang generated
        // functions (because Clang annotates functions this way too).
        self.apply_target_cpu_attr(fun);
    }

    fn apply_optimization_attributes(&self, fun: Value) {
        match self.opt_level {
            PassBuilderOptLevel::O0 => {
                self.remove_function_attr(fun, Attribute::MinSize);
                self.remove_function_attr(fun, Attribute::OptimizeForSize);
                self.set_function_attr(fun, Attribute::OptimizeNone);
            }
            PassBuilderOptLevel::Os => {
                self.remove_function_attr(fun, Attribute::MinSize);
                self.set_function_attr(fun, Attribute::OptimizeForSize);
                self.remove_function_attr(fun, Attribute::OptimizeNone);
            }
            PassBuilderOptLevel::Oz => {
                self.set_function_attr(fun, Attribute::MinSize);
                self.set_function_attr(fun, Attribute::OptimizeForSize);
                self.remove_function_attr(fun, Attribute::OptimizeNone);
            }
            _ => {
                self.remove_function_attr(fun, Attribute::MinSize);
                self.remove_function_attr(fun, Attribute::OptimizeForSize);
                self.remove_function_attr(fun, Attribute::OptimizeNone);
            }
        }
    }

    fn apply_sanitizers(&self, fun: Value) {
        if let Some(sanitizer) = self.sanitizer {
            match sanitizer {
                Sanitizer::Address => {
                    self.set_function_attr(fun, Attribute::SanitizeAddress);
                }
                Sanitizer::Memory => {
                    self.set_function_attr(fun, Attribute::SanitizeMemory);
                }
                Sanitizer::Thread => {
                    self.set_function_attr(fun, Attribute::SanitizeThread);
                }
                Sanitizer::Leak => {}
            }
        }
    }

    fn apply_target_cpu_attr(&self, fun: Value) {
        let target_cpu_name = CString::new(self.target_cpu.as_str()).unwrap();
        self.set_function_attr_string_value(
            fun,
            AttributePlace::Function,
            TARGET_CPU_STR,
            target_cpu_name.as_c_str(),
        );
    }

    pub fn get_intrinsic(&self, key: &str) -> Value {
        if let Some(v) = self.intrinsics.borrow().get(key).cloned() {
            return v;
        }

        self.declare_intrinsic(key)
            .unwrap_or_else(|| panic!("unknown intrinsic '{}'", key))
    }

    fn insert_intrinsic(&self, name: &'static str, args: Option<&[Type]>, ret: Type) -> Value {
        use llvm_sys::core::LLVMSetUnnamedAddress;
        use llvm_sys::LLVMUnnamedAddr;

        let fn_ty = if let Some(args) = args {
            self.get_function_type(ret, args, /* variadic */ false)
        } else {
            self.get_function_type(ret, &[], /* variadic */ true)
        };
        let f = self.declare_function(name, fn_ty);
        unsafe { LLVMSetUnnamedAddress(f, LLVMUnnamedAddr::LLVMNoUnnamedAddr) };
        self.intrinsics.borrow_mut().insert(name, f);
        f
    }

    fn declare_intrinsic(&self, key: &str) -> Option<Value> {
        macro_rules! ifn {
            ($name:expr, fn() -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, Some(&[]), $ret));
                }
            );
            ($name:expr, fn(...) -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, None, $ret));
                }
            );
            ($name:expr, fn($($arg:expr),*) -> $ret:expr) => (
                if key == $name {
                    return Some(self.insert_intrinsic($name, Some(&[$($arg),*]), $ret));
                }
            );
        }
        macro_rules! mk_struct {
            ($($field_ty:expr),*) => (self.get_struct_type(None, &[$($field_ty),*]))
        }

        let t_i8 = self.get_i8_type();
        let i8p = self.get_pointer_type(t_i8);
        let void = self.get_void_type();
        let i1 = self.get_integer_type(1);
        let t_i16 = self.get_integer_type(16);
        let t_i32 = self.get_i32_type();
        let t_i64 = self.get_i64_type();
        let t_i128 = self.get_integer_type(128);
        let t_f32 = self.get_f32_type();
        let t_f64 = self.get_f64_type();
        let t_meta = self.get_metadata_type();
        let t_token = self.get_token_type();

        macro_rules! vector_types {
            ($id_out:ident: $elem_ty:ident, $len:expr) => {
                let $id_out = self.get_vector_type($elem_ty, $len);
            };
            ($($id_out:ident: $elem_ty:ident, $len:expr;)*) => {
                $(vector_types!($id_out: $elem_ty, $len);)*
            }
        }
        vector_types! {
            t_v2f32: t_f32, 2;
            t_v4f32: t_f32, 4;
            t_v8f32: t_f32, 8;
            t_v16f32: t_f32, 16;

            t_v2f64: t_f64, 2;
            t_v4f64: t_f64, 4;
            t_v8f64: t_f64, 8;
        }

        ifn!("llvm.trap", fn() -> void);
        ifn!("llvm.debugtrap", fn() -> void);
        ifn!("llvm.frameaddress", fn(t_i32) -> i8p);
        ifn!("llvm.sideeffect", fn() -> void);

        ifn!("llvm.powi.f32", fn(t_f32, t_i32) -> t_f32);
        ifn!("llvm.powi.v2f32", fn(t_v2f32, t_i32) -> t_v2f32);
        ifn!("llvm.powi.v4f32", fn(t_v4f32, t_i32) -> t_v4f32);
        ifn!("llvm.powi.v8f32", fn(t_v8f32, t_i32) -> t_v8f32);
        ifn!("llvm.powi.v16f32", fn(t_v16f32, t_i32) -> t_v16f32);
        ifn!("llvm.powi.f64", fn(t_f64, t_i32) -> t_f64);
        ifn!("llvm.powi.v2f64", fn(t_v2f64, t_i32) -> t_v2f64);
        ifn!("llvm.powi.v4f64", fn(t_v4f64, t_i32) -> t_v4f64);
        ifn!("llvm.powi.v8f64", fn(t_v8f64, t_i32) -> t_v8f64);

        ifn!("llvm.pow.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.pow.v2f32", fn(t_v2f32, t_v2f32) -> t_v2f32);
        ifn!("llvm.pow.v4f32", fn(t_v4f32, t_v4f32) -> t_v4f32);
        ifn!("llvm.pow.v8f32", fn(t_v8f32, t_v8f32) -> t_v8f32);
        ifn!("llvm.pow.v16f32", fn(t_v16f32, t_v16f32) -> t_v16f32);
        ifn!("llvm.pow.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.pow.v2f64", fn(t_v2f64, t_v2f64) -> t_v2f64);
        ifn!("llvm.pow.v4f64", fn(t_v4f64, t_v4f64) -> t_v4f64);
        ifn!("llvm.pow.v8f64", fn(t_v8f64, t_v8f64) -> t_v8f64);

        ifn!("llvm.sqrt.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sqrt.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.sqrt.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.sqrt.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.sqrt.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.sqrt.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sqrt.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.sqrt.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.sqrt.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.sin.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sin.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.sin.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.sin.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.sin.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.sin.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.sin.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.sin.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.sin.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.cos.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.cos.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.cos.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.cos.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.cos.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.cos.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.cos.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.cos.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.cos.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.exp.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.exp.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.exp.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.exp.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.exp.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.exp.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.exp.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.exp2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp2.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.exp2.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.exp2.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.exp2.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.exp2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.exp2.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.exp2.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.exp2.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log10.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log10.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log10.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log10.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log10.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log10.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log10.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log10.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log10.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.log2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log2.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.log2.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.log2.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.log2.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.log2.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.log2.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.log2.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.log2.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.fma.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
        ifn!("llvm.fma.v2f32", fn(t_v2f32, t_v2f32, t_v2f32) -> t_v2f32);
        ifn!("llvm.fma.v4f32", fn(t_v4f32, t_v4f32, t_v4f32) -> t_v4f32);
        ifn!("llvm.fma.v8f32", fn(t_v8f32, t_v8f32, t_v8f32) -> t_v8f32);
        ifn!(
            "llvm.fma.v16f32",
            fn(t_v16f32, t_v16f32, t_v16f32) -> t_v16f32
        );
        ifn!("llvm.fma.f64", fn(t_f64, t_f64, t_f64) -> t_f64);
        ifn!("llvm.fma.v2f64", fn(t_v2f64, t_v2f64, t_v2f64) -> t_v2f64);
        ifn!("llvm.fma.v4f64", fn(t_v4f64, t_v4f64, t_v4f64) -> t_v4f64);
        ifn!("llvm.fma.v8f64", fn(t_v8f64, t_v8f64, t_v8f64) -> t_v8f64);

        ifn!("llvm.fabs.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.fabs.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.fabs.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.fabs.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.fabs.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.fabs.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.fabs.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.fabs.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.fabs.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.minnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.minnum.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.maxnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.maxnum.f64", fn(t_f64, t_f64) -> t_f64);

        ifn!("llvm.floor.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.floor.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.floor.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.floor.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.floor.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.floor.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.floor.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.floor.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.floor.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.ceil.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.ceil.v2f32", fn(t_v2f32) -> t_v2f32);
        ifn!("llvm.ceil.v4f32", fn(t_v4f32) -> t_v4f32);
        ifn!("llvm.ceil.v8f32", fn(t_v8f32) -> t_v8f32);
        ifn!("llvm.ceil.v16f32", fn(t_v16f32) -> t_v16f32);
        ifn!("llvm.ceil.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.ceil.v2f64", fn(t_v2f64) -> t_v2f64);
        ifn!("llvm.ceil.v4f64", fn(t_v4f64) -> t_v4f64);
        ifn!("llvm.ceil.v8f64", fn(t_v8f64) -> t_v8f64);

        ifn!("llvm.trunc.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.trunc.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.copysign.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.copysign.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.round.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.round.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.rint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.rint.f64", fn(t_f64) -> t_f64);
        ifn!("llvm.nearbyint.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.nearbyint.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.ctpop.i8", fn(t_i8) -> t_i8);
        ifn!("llvm.ctpop.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.ctpop.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.ctpop.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.ctpop.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.ctlz.i8", fn(t_i8, i1) -> t_i8);
        ifn!("llvm.ctlz.i16", fn(t_i16, i1) -> t_i16);
        ifn!("llvm.ctlz.i32", fn(t_i32, i1) -> t_i32);
        ifn!("llvm.ctlz.i64", fn(t_i64, i1) -> t_i64);
        ifn!("llvm.ctlz.i128", fn(t_i128, i1) -> t_i128);

        ifn!("llvm.cttz.i8", fn(t_i8, i1) -> t_i8);
        ifn!("llvm.cttz.i16", fn(t_i16, i1) -> t_i16);
        ifn!("llvm.cttz.i32", fn(t_i32, i1) -> t_i32);
        ifn!("llvm.cttz.i64", fn(t_i64, i1) -> t_i64);
        ifn!("llvm.cttz.i128", fn(t_i128, i1) -> t_i128);

        ifn!("llvm.bswap.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.bswap.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.bswap.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.bswap.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.bitreverse.i8", fn(t_i8) -> t_i8);
        ifn!("llvm.bitreverse.i16", fn(t_i16) -> t_i16);
        ifn!("llvm.bitreverse.i32", fn(t_i32) -> t_i32);
        ifn!("llvm.bitreverse.i64", fn(t_i64) -> t_i64);
        ifn!("llvm.bitreverse.i128", fn(t_i128) -> t_i128);

        ifn!("llvm.fshl.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!("llvm.fshl.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!("llvm.fshl.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!("llvm.fshl.i64", fn(t_i64, t_i64, t_i64) -> t_i64);
        ifn!("llvm.fshl.i128", fn(t_i128, t_i128, t_i128) -> t_i128);

        ifn!("llvm.fshr.i8", fn(t_i8, t_i8, t_i8) -> t_i8);
        ifn!("llvm.fshr.i16", fn(t_i16, t_i16, t_i16) -> t_i16);
        ifn!("llvm.fshr.i32", fn(t_i32, t_i32, t_i32) -> t_i32);
        ifn!("llvm.fshr.i64", fn(t_i64, t_i64, t_i64) -> t_i64);
        ifn!("llvm.fshr.i128", fn(t_i128, t_i128, t_i128) -> t_i128);

        ifn!(
            "llvm.sadd.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.sadd.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.sadd.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.sadd.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.sadd.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!(
            "llvm.uadd.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.uadd.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.uadd.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.uadd.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.uadd.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!(
            "llvm.ssub.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.ssub.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.ssub.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.ssub.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.ssub.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!(
            "llvm.usub.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.usub.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.usub.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.usub.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.usub.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!(
            "llvm.smul.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.smul.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.smul.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.smul.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.smul.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!(
            "llvm.umul.with.overflow.i8",
            fn(t_i8, t_i8) -> mk_struct! {t_i8, i1}
        );
        ifn!(
            "llvm.umul.with.overflow.i16",
            fn(t_i16, t_i16) -> mk_struct! {t_i16, i1}
        );
        ifn!(
            "llvm.umul.with.overflow.i32",
            fn(t_i32, t_i32) -> mk_struct! {t_i32, i1}
        );
        ifn!(
            "llvm.umul.with.overflow.i64",
            fn(t_i64, t_i64) -> mk_struct! {t_i64, i1}
        );
        ifn!(
            "llvm.umul.with.overflow.i128",
            fn(t_i128, t_i128) -> mk_struct! {t_i128, i1}
        );

        ifn!("llvm.sadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.sadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.sadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.sadd.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.sadd.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.uadd.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.uadd.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.uadd.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.uadd.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.uadd.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.ssub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.ssub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.ssub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.ssub.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.ssub.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.usub.sat.i8", fn(t_i8, t_i8) -> t_i8);
        ifn!("llvm.usub.sat.i16", fn(t_i16, t_i16) -> t_i16);
        ifn!("llvm.usub.sat.i32", fn(t_i32, t_i32) -> t_i32);
        ifn!("llvm.usub.sat.i64", fn(t_i64, t_i64) -> t_i64);
        ifn!("llvm.usub.sat.i128", fn(t_i128, t_i128) -> t_i128);

        ifn!("llvm.lifetime.start.p0i8", fn(t_i64, i8p) -> void);
        ifn!("llvm.lifetime.end.p0i8", fn(t_i64, i8p) -> void);

        ifn!("llvm.expect.i1", fn(i1, i1) -> i1);
        ifn!("llvm.eh.typeid.for", fn(i8p) -> t_i32);
        ifn!("llvm.localescape", fn(...) -> void);
        ifn!("llvm.localrecover", fn(i8p, i8p, t_i32) -> i8p);
        ifn!("llvm.x86.seh.recoverfp", fn(i8p, i8p) -> i8p);

        ifn!("llvm.assume", fn(i1) -> void);
        ifn!("llvm.prefetch", fn(i8p, t_i32, t_i32, t_i32) -> void);

        // variadic intrinsics
        ifn!("llvm.va_start", fn(i8p) -> void);
        ifn!("llvm.va_end", fn(i8p) -> void);
        ifn!("llvm.va_copy", fn(i8p, i8p) -> void);

        ifn!("llvm.dbg.declare", fn(t_meta, t_meta) -> void);
        ifn!("llvm.dbg.value", fn(t_meta, t_i64, t_meta) -> void);

        // wasm32 exception handling intrinsics
        ifn!("llvm.wasm.get.exception", fn(t_token) -> i8p);
        ifn!("llvm.wasm.get.ehselector", fn(t_token) -> t_i32);

        None
    }
}

const BASE_N_ALPHANUMERIC_ONLY: usize = 62;
// const BASE_N_CASE_INSENSITIVE: usize = 36;

#[inline]
fn push_base_n_str(mut n: u128, base: usize, output: &mut String) {
    const MAX_BASE: usize = 64;
    const BASE_64: &[u8; MAX_BASE as usize] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ@$";

    debug_assert!(base >= 2 && base <= MAX_BASE);
    let mut s = [0u8; 128];
    let mut index = 0;

    let base = base as u128;

    loop {
        s[index] = BASE_64[(n % base) as usize];
        index += 1;
        n /= base;

        if n == 0 {
            break;
        }
    }
    s[0..index].reverse();

    output.push_str(std::str::from_utf8(&s[0..index]).unwrap());
}

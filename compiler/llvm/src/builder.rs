use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::ops::Deref;

use fxhash::FxHashMap;

use liblumen_session::{Options, Sanitizer};

use crate::codegen;
use crate::ir::*;
use crate::passes::PassBuilderOptLevel;
use crate::support::*;
use crate::target::*;

extern "C" {
    type LlvmBuilder;
}

/// Represents a borrowed reference to an LLVM builder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Builder(*const LlvmBuilder);
impl Builder {}

/// Represents an owned reference to an LLVM builder
#[repr(transparent)]
pub struct OwnedBuilder(Builder);
impl OwnedBuilder {
    pub fn new(context: Context) -> Self {
        extern "C" {
            fn LLVMCreateBuilderInContext(context: Context) -> OwnedBuilder;
        }
        unsafe { LLVMCreateBuilderInContext(context) }
    }
}
impl Deref for OwnedBuilder {
    type Target = Builder;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedBuilder {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMDisposeBuilder(builder: Builder);
        }
        unsafe { LLVMDisposeBuilder(self.0) }
    }
}

/// Empty string, to be used where LLVM expects an instruction name, indicating
/// that the instruction is to be left unnamed (i.e. numbered, in textual IR).
const EMPTY_C_STR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(b"\0") };
const UNNAMED: *const std::os::raw::c_char = EMPTY_C_STR.as_ptr();

/// A module builder extends an LLVM builder for use in defining LLVM IR modules
pub struct ModuleBuilder<'ctx> {
    builder: OwnedBuilder,
    context: &'ctx Context,
    module: OwnedModule,
    intrinsics: RefCell<FxHashMap<&'static str, Function>>,
    local_gen_sym_counter: Cell<usize>,
    opt_level: PassBuilderOptLevel,
    sanitizer: Option<Sanitizer>,
    target_cpu: String,
    usize_type: IntegerType,
}
impl<'ctx> ModuleBuilder<'ctx> {
    pub fn new(
        name: &str,
        options: &Options,
        context: &'ctx Context,
        target_machine: TargetMachine,
    ) -> anyhow::Result<Self> {
        // Initialize the module
        let module = context.create_module(name);
        module.set_target_triple(target_machine.triple());
        let data_layout = target_machine.data_layout();
        let usize_type = data_layout.get_int_ptr_type(*context);
        module.set_data_layout(data_layout);

        let builder = context.create_builder();

        let (speed, size) = codegen::to_llvm_opt_settings(options.opt_level);
        let opt_level = PassBuilderOptLevel::from_codegen_opts(speed, size);

        Ok(Self {
            builder,
            context,
            module,
            intrinsics: RefCell::new(Default::default()),
            local_gen_sym_counter: Cell::new(0),
            opt_level,
            sanitizer: options.debugging_opts.sanitizer.clone(),
            target_cpu: crate::target::target_cpu(options).to_owned(),
            usize_type,
        })
    }

    pub fn finish(self) -> anyhow::Result<OwnedModule> {
        Ok(self.module)
    }

    /// Build an entry block for the given function
    pub fn build_entry_block(&self, fun: Function) -> Block {
        self.build_named_block(fun, "entry")
    }

    /// Build an anonymous block for the given function, appending it to the end
    pub fn build_block(&self, fun: Function) -> Block {
        self.build_block_with_name(fun, UNNAMED)
    }

    /// Build a named block for the given function, appending it to the end
    pub fn build_named_block<S: Into<StringRef>>(&self, fun: Function, name: S) -> Block {
        let name = name.into();
        let c_str = name.to_cstr();
        self.build_block_with_name(fun, c_str.as_ptr())
    }

    fn build_block_with_name(&self, fun: Function, name: *const std::os::raw::c_char) -> Block {
        extern "C" {
            fn LLVMAppendBasicBlockInContext(
                context: Context,
                fun: Function,
                name: *const std::os::raw::c_char,
            ) -> Block;
        }

        unsafe { LLVMAppendBasicBlockInContext(*self.context, fun, name) }
    }

    /// Returns the current block the builder is inserting in
    pub fn current_block(&self) -> Block {
        extern "C" {
            fn LLVMGetInsertBlock(builder: Builder) -> Block;
        }
        unsafe { LLVMGetInsertBlock(*self.builder) }
    }

    /// Position the builder at the end of `block`
    pub fn position_at_end(&self, block: Block) {
        extern "C" {
            fn LLVMPositionBuilderAtEnd(builder: Builder, block: Block);
        }

        unsafe {
            LLVMPositionBuilderAtEnd(*self.builder, block);
        }
    }

    /// Position the builder just before `inst`
    pub fn position_before(&self, inst: ValueBase) {
        extern "C" {
            fn LLVMPositionBuilderBefore(builder: Builder, inst: ValueBase);
        }
        unsafe {
            LLVMPositionBuilderBefore(*self.builder, inst);
        }
    }

    /// Inserts the given instruction at the builder's current position
    pub fn insert<I: Instruction>(&self, inst: I) {
        extern "C" {
            fn LLVMInsertIntoBuilder(builder: Builder, inst: ValueBase);
        }
        unsafe { LLVMInsertIntoBuilder(*self.builder, inst.base()) }
    }

    /// Returns the type representing void/noreturn
    pub fn get_void_type(&self) -> VoidType {
        self.context.get_void_type()
    }

    /// Returns a type representing a 1-bit wide integer/bool
    pub fn get_i1_type(&self) -> IntegerType {
        self.context.get_i1_type()
    }

    /// Returns a type representing an 8-bit wide integer/char
    pub fn get_i8_type(&self) -> IntegerType {
        self.context.get_i8_type()
    }

    /// Returns a type representing a 16-bit wide integer/short
    pub fn get_i16_type(&self) -> IntegerType {
        self.context.get_i16_type()
    }

    /// Returns a type representing a 32-bit wide integer/long
    pub fn get_i32_type(&self) -> IntegerType {
        self.context.get_i32_type()
    }

    /// Returns a type representing a 64-bit wide integer/longlong
    pub fn get_i64_type(&self) -> IntegerType {
        self.context.get_i64_type()
    }

    /// Returns a type representing a 128-bit wide integer
    pub fn get_i128_type(&self) -> IntegerType {
        self.context.get_i28_type()
    }

    /// Returns a type representing a pointer-width integer value for the current target
    pub fn get_usize_type(&self) -> IntegerType {
        self.usize_type
    }

    /// Returns a type representing an integer which is `width` bits wide
    pub fn get_integer_type(&self, width: usize) -> IntegerType {
        self.context.get_integer_type(width)
    }

    /// Returns a type representing a 32-bit floating point value
    pub fn get_f32_type(&self) -> FloatType {
        self.context.get_f32_type()
    }

    /// Returns a type representing a 64-bit floating point value
    pub fn get_f64_type(&self) -> FloatType {
        self.context.get_f64_type()
    }

    /// Returns a type used for LLVM metadata
    pub fn get_metadata_type(&self) -> MetadataType {
        self.context.get_metadata_type()
    }

    /// Returns a type used for LLVM token values
    pub fn get_token_type(&self) -> TokenType {
        self.context.get_token_type()
    }

    /// Returns an array type of the given size and element type
    pub fn get_array_type<T: Type>(&self, size: usize, ty: T) -> ArrayType {
        ArrayType::new(ty, size)
    }

    /// Returns a struct type, optionally named, with the given set of fields
    pub fn get_struct_type(&self, name: Option<&str>, field_types: &[TypeBase]) -> StructType {
        if let Some(name) = name {
            self.context
                .get_named_struct_type(name, field_types, /*packed=*/ false)
        } else {
            self.context.get_struct_type(field_types)
        }
    }

    /// Returns a pointer type, with the given pointee type
    pub fn get_pointer_type<T: Type>(&self, ty: T) -> PointerType {
        PointerType::new(ty, 0)
    }

    /// Returns a type corresponding to a function with the given return value and argument types, optionally variadic
    pub fn get_function_type<T: Type>(
        &self,
        ret: T,
        params: &[TypeBase],
        variadic: bool,
    ) -> FunctionType {
        FunctionType::new(ret, params, variadic)
    }

    /// A helper function which returns a function type consisting of all terms, with the given number of arguments
    pub fn get_erlang_function_type(&self, arity: u8) -> FunctionType {
        let term_type = self.get_term_type().base();
        let arity = arity as usize;
        let mut params = Vec::with_capacity(arity);
        params.resize(arity, term_type);
        self.get_function_type(term_type, params.as_slice(), /* variadic */ false)
    }

    /// Returns a type corresponding to `typedef void (*fun)()`, or a function pointer
    /// which has no defined arguments/return value. Such a type is intended to be cast
    /// to a more concrete function type depending on runtime conditions.
    pub fn get_opaque_function_type(&self) -> FunctionType {
        let void_type = self.get_void_type();
        self.get_function_type(void_type, &[], /* variadic */ false)
    }

    /// Returns an appropriate LLVM type which represents an opaque Erlang term (i.e. immediate)
    pub fn get_term_type(&self) -> IntegerType {
        self.usize_type
    }

    /// Get the current location used by debug info produced by this builder
    pub fn current_debug_location(&self) -> Metadata {
        extern "C" {
            fn LLVMGetCurrentDebugLocation2(builder: Builder) -> Metadata;
        }
        unsafe { LLVMGetCurrentDebugLocation2(*self.builder) }
    }

    /// Set the location to use for debug info generated by this builder
    ///
    /// NOTE: To clear location metadata, pass `Metadata::null()`
    pub fn set_current_debug_location(&self, loc: Metadata) {
        extern "C" {
            fn LLVMSetCurrentDebugLocation2(builder: Builder, loc: Metadata);
        }
        unsafe { LLVMSetCurrentDebugLocation2(*self.builder, loc) }
    }

    /// Sets the debug location for the given instruction using the builder's current debug location metadata
    pub fn set_debug_location<I: Instruction>(&self, inst: I) {
        extern "C" {
            fn LLVMSetInstDebugLocation(builder: Builder, inst: ValueBase);
        }
        unsafe {
            LLVMSetInstDebugLocation(*self.builder, inst.base());
        }
    }

    /// Sets the default floating-point math metadata for the current builder
    ///
    /// NOTE: To clear the default, pass `Metadata::null()`
    pub fn set_default_fp_math_tag(&self, tag: Metadata) {
        extern "C" {
            fn LLVMBuilderSetDefaultFPMathTag(builder: Builder, tag: Metadata);
        }
        unsafe {
            LLVMBuilderSetDefaultFPMathTag(*self.builder, tag);
        }
    }

    pub fn build_is_null<V: Value>(&self, value: V) -> ValueBase {
        extern "C" {
            fn LLVMBuildIsNull(
                builder: Builder,
                value: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildIsNull(*self.builder, value.base(), UNNAMED) }
    }

    pub fn build_bitcast<V: Value, T: Type>(&self, value: V, ty: T) -> ValueBase {
        extern "C" {
            fn LLVMBuildBitCast(
                builder: Builder,
                value: ValueBase,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildBitCast(*self.builder, value.base(), ty.base(), UNNAMED) }
    }

    pub fn build_inttoptr<V: Value, T: Type>(&self, value: V, ty: T) -> ValueBase {
        extern "C" {
            fn LLVMBuildIntToPtr(
                builder: Builder,
                value: ValueBase,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildIntToPtr(*self.builder, value.base(), ty.base(), UNNAMED) }
    }

    pub fn build_ptrtoint<V: Value>(&self, value: V, ty: IntegerType) -> ValueBase {
        extern "C" {
            fn LLVMBuildPtrToInt(
                builder: Builder,
                value: ValueBase,
                ty: IntegerType,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildPtrToInt(*self.builder, value.base(), ty, UNNAMED) }
    }

    pub fn build_and<L, R>(&self, lhs: L, rhs: R) -> ValueBase
    where
        L: Value,
        R: Value,
    {
        extern "C" {
            fn LLVMBuildAnd(
                builder: Builder,
                lhs: ValueBase,
                rhs: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildAnd(*self.builder, lhs.base(), rhs.base(), UNNAMED) }
    }

    pub fn build_or<L, R>(&self, lhs: L, rhs: R) -> ValueBase
    where
        L: Value,
        R: Value,
    {
        extern "C" {
            fn LLVMBuildOr(
                builder: Builder,
                lhs: ValueBase,
                rhs: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildOr(*self.builder, lhs.base(), rhs.base(), UNNAMED) }
    }

    pub fn build_xor<L, R>(&self, lhs: L, rhs: R) -> ValueBase
    where
        L: Value,
        R: Value,
    {
        extern "C" {
            fn LLVMBuildXor(
                builder: Builder,
                lhs: ValueBase,
                rhs: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildXor(*self.builder, lhs.base(), rhs.base(), UNNAMED) }
    }

    pub fn build_not<V: Value>(&self, value: V) -> ValueBase {
        extern "C" {
            fn LLVMBuildNot(
                builder: Builder,
                value: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        unsafe { LLVMBuildNot(*self.builder, value.base(), UNNAMED) }
    }

    pub fn build_undef<T: Type>(&self, ty: T) -> UndefValue {
        UndefValue::get(ty)
    }

    pub fn build_constant_null<T: Type>(&self, ty: T) -> ConstantValue {
        ConstantValue::null(ty)
    }

    pub fn build_constant_int(&self, ty: IntegerType, value: i64) -> ConstantInt {
        ConstantInt::get(ty, value as u64, /*sext=*/ true)
    }

    pub fn build_constant_uint(&self, ty: IntegerType, value: u64) -> ConstantInt {
        ConstantInt::get(ty, value, /*sext=*/ false)
    }

    pub fn build_constant_array<T: Type>(&self, ty: T, values: &[ConstantValue]) -> ConstantArray {
        ConstantArray::get(ty, values)
    }

    pub fn build_constant_bytes(&self, bytes: &[u8]) -> ConstantString {
        self.context.const_string(bytes)
    }

    pub fn build_constant_string<S: Into<StringRef>>(&self, s: S) -> GlobalVariable {
        let sym = self.generate_local_symbol_name("str");

        self.build_named_constant_string(sym.as_str(), s)
    }

    pub fn build_named_constant_string<N: Into<StringRef>, S: Into<StringRef>>(
        &self,
        name: N,
        s: S,
    ) -> GlobalVariable {
        let value = self.context.const_string(s);

        let name = name.into();
        let gv = self
            .define_global(name, value.get_type())
            .unwrap_or_else(|| {
                panic!("symbol `{}` is already defined", name);
            });
        gv.set_initializer(value);
        gv.set_linkage(Linkage::Internal);
        gv
    }

    pub fn build_constant_named_struct(
        &self,
        ty: StructType,
        fields: &[ConstantValue],
    ) -> ConstantStruct {
        ConstantStruct::get_named(ty, fields)
    }

    pub fn build_constant_unnamed_struct(&self, elements: &[ConstantValue]) -> ConstantStruct {
        self.context.const_struct(elements)
    }

    pub fn build_constant_get_value<A: ConstantAggregate>(
        &self,
        agg: A,
        index: u64,
    ) -> ConstantExpr {
        let index_ty = self.context.get_i32_type();
        let index = ConstantInt::get(index_ty, index, /*sext=*/ false);
        ConstantExpr::extract_value(agg, &[index.into()])
    }

    pub fn declare_function<S: Into<StringRef>>(&self, name: S, ty: FunctionType) -> Function {
        let fun = self.module.get_or_add_function(name, ty);

        self.apply_default_function_attributes(fun);

        fun
    }

    #[inline]
    pub fn build_function<S: Into<StringRef>>(&self, name: S, ty: FunctionType) -> Function {
        self.build_function_with_attrs(name, ty, Linkage::Internal, &[])
    }

    #[inline]
    pub fn build_external_function<S: Into<StringRef>>(
        &self,
        name: S,
        ty: FunctionType,
    ) -> Function {
        self.build_function_with_attrs(name, ty, Linkage::External, &[])
    }

    pub fn build_function_with_attrs<S: Into<StringRef>>(
        &self,
        name: S,
        ty: FunctionType,
        linkage: Linkage,
        attrs: &[AttributeBase],
    ) -> Function {
        let fun = self.declare_function(name, ty);
        fun.set_linkage(linkage);

        for attr in attrs.iter().copied() {
            fun.add_attribute(attr);
        }

        fun
    }

    pub fn build_alloca<T: Type>(&self, ty: T) -> AllocaInst {
        extern "C" {
            fn LLVMBuildAlloca(
                builder: Builder,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> AllocaInst;
        }

        unsafe { LLVMBuildAlloca(*self.builder, ty.base(), UNNAMED) }
    }

    pub fn build_array_alloca<T: Type>(&self, ty: T, arity: ConstantValue) -> AllocaInst {
        extern "C" {
            fn LLVMBuildArrayAlloca(
                builder: Builder,
                ty: TypeBase,
                arity: ConstantValue,
                name: *const std::os::raw::c_char,
            ) -> AllocaInst;
        }

        unsafe { LLVMBuildArrayAlloca(*self.builder, ty.base(), arity, UNNAMED) }
    }

    pub fn build_malloc<T: Type>(&self, ty: T) -> CallInst {
        extern "C" {
            fn LLVMBuildMalloc(
                builder: Builder,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> CallInst;
        }

        unsafe { LLVMBuildMalloc(*self.builder, ty.base(), UNNAMED) }
    }

    pub fn build_array_malloc<T: Type>(&self, ty: T, arity: ConstantValue) -> CallInst {
        extern "C" {
            fn LLVMBuildArrayMalloc(
                builder: Builder,
                ty: TypeBase,
                arity: ConstantValue,
                name: *const std::os::raw::c_char,
            ) -> CallInst;
        }

        unsafe { LLVMBuildArrayMalloc(*self.builder, ty.base(), arity, UNNAMED) }
    }

    pub fn build_free<V: Value>(&self, ptr: V) -> CallInst {
        extern "C" {
            fn LLVMBuildFree(builder: Builder, pointer: ValueBase) -> CallInst;
        }
        unsafe { LLVMBuildFree(*self.builder, ptr.base()) }
    }

    pub fn build_phi<T: Type>(&self, ty: T, incoming: &[(ValueBase, Block)]) -> PhiInst {
        extern "C" {
            fn LLVMBuildPhi(
                builder: Builder,
                ty: TypeBase,
                name: *const std::os::raw::c_char,
            ) -> PhiInst;
        }

        let phi = unsafe { LLVMBuildPhi(*self.builder, ty.base(), UNNAMED) };
        phi.add_incoming(incoming);

        phi
    }

    pub fn build_br(&self, dest: Block) -> BranchInst {
        extern "C" {
            fn LLVMBuildBr(builder: Builder, dest: Block) -> BranchInst;
        }

        unsafe { LLVMBuildBr(*self.builder, dest) }
    }

    pub fn build_condbr(&self, cond: ValueBase, then_dest: Block, else_dest: Block) -> BranchInst {
        extern "C" {
            fn LLVMBuildCondBr(
                builder: Builder,
                cond: ValueBase,
                true_dest: Block,
                false_dest: Block,
            ) -> BranchInst;
        }

        unsafe { LLVMBuildCondBr(*self.builder, cond, then_dest, else_dest) }
    }

    pub fn build_switch(&self, value: ValueBase, default: Block) -> SwitchInst {
        extern "C" {
            fn LLVMBuildSwitch(
                builder: Builder,
                value: ValueBase,
                default: Block,
                num_cases_hint: u32,
            ) -> SwitchInst;
        }

        unsafe {
            LLVMBuildSwitch(*self.builder, value, default, /*hint=*/ 10)
        }
    }

    pub fn build_icmp<L, R>(&self, lhs: L, rhs: R, predicate: ICmp) -> ICmpInst
    where
        L: Value,
        R: Value,
    {
        extern "C" {
            fn LLVMBuildICmp(
                builder: Builder,
                pred: ICmp,
                lhs: ValueBase,
                rhs: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> ICmpInst;
        }

        unsafe { LLVMBuildICmp(*self.builder, predicate, lhs.base(), rhs.base(), UNNAMED) }
    }

    /// Builds a call instruction with a statically known function as callee
    pub fn build_call(
        &self,
        fun: Function,
        args: &[ValueBase],
        funclet: Option<&Funclet>,
    ) -> CallInst {
        let ty = fun.get_type().try_into().unwrap();
        self.build_call_indirect(fun.base(), ty, args, funclet)
    }

    /// Builds a call instruction with a callee value that may or may not be a function reference
    ///
    /// The primary difference with `build_invoke` is that this requires providing the type of the callee
    pub fn build_call_indirect(
        &self,
        callee: ValueBase,
        callee_type: FunctionType,
        args: &[ValueBase],
        funclet: Option<&Funclet>,
    ) -> CallInst {
        extern "C" {
            fn LLVMLumenBuildCall(
                builder: Builder,
                callee: ValueBase,
                callee_type: FunctionType,
                argv: *const ValueBase,
                argc: u32,
                bundle: OperandBundle,
                name: *const std::os::raw::c_char,
            ) -> CallInst;
        }

        let argv = args.as_ptr() as *mut _;
        let argc = args.len().try_into().unwrap();
        let funclet = funclet
            .map(|f| f.bundle())
            .unwrap_or_else(OperandBundle::null);

        unsafe {
            LLVMLumenBuildCall(
                *self.builder,
                callee,
                callee_type,
                argv,
                argc,
                funclet,
                UNNAMED,
            )
        }
    }

    /// Builds an invoke instruction with a statically known function as callee
    pub fn build_invoke(
        &self,
        fun: Function,
        args: &[ValueBase],
        normal: Block,
        unwind: Block,
        funclet: Option<&Funclet>,
    ) -> InvokeInst {
        let ty = fun.get_type().try_into().unwrap();
        self.build_invoke_indirect(fun.base(), ty, args, normal, unwind, funclet)
    }

    /// Builds an invoke instruction with a callee value that may or may not be a function reference
    ///
    /// The primary difference with `build_invoke` is that this requires providing the type of the callee
    pub fn build_invoke_indirect(
        &self,
        callee: ValueBase,
        callee_type: FunctionType,
        args: &[ValueBase],
        normal: Block,
        unwind: Block,
        funclet: Option<&Funclet>,
    ) -> InvokeInst {
        extern "C" {
            fn LLVMLumenBuildInvoke(
                builder: Builder,
                callee: ValueBase,
                callee_type: FunctionType,
                argv: *const ValueBase,
                argc: u32,
                ok: Block,
                catch: Block,
                bundle: OperandBundle,
                name: *const std::os::raw::c_char,
            ) -> InvokeInst;
        }

        let argv = args.as_ptr();
        let argc = args.len().try_into().unwrap();
        let funclet = funclet
            .map(|f| f.bundle())
            .unwrap_or_else(OperandBundle::null);

        unsafe {
            LLVMLumenBuildInvoke(
                *self.builder,
                callee,
                callee_type,
                argv,
                argc,
                normal,
                unwind,
                funclet,
                UNNAMED,
            )
        }
    }

    pub fn build_return<V: Value>(&self, ret: V) -> ReturnInst {
        extern "C" {
            fn LLVMBuildRet(builder: Builder, value: ValueBase) -> ReturnInst;
        }

        unsafe { LLVMBuildRet(*self.builder, ret.base()) }
    }

    pub fn build_unreachable(&self) -> UnreachableInst {
        extern "C" {
            fn LLVMBuildUnreachable(builder: Builder) -> UnreachableInst;
        }

        unsafe { LLVMBuildUnreachable(*self.builder) }
    }

    /// Resumes propagation of the given exception.
    ///
    /// Part of the set of older Itanium C++ exception handling instructions
    pub fn build_resume<V: Value>(&self, exception: V) -> ResumeInst {
        extern "C" {
            fn LLVMBuildResume(builder: Builder, exception: ValueBase) -> ResumeInst;
        }

        unsafe { LLVMBuildResume(*self.builder, exception.base()) }
    }

    /// When inserted at the beginning of a block, it marks the block as a catch or cleanup handler
    /// for a matching `invoke` instruction. When the callee of an invoke unwinds, the landingpad, if
    /// matching, will be visited. Unwinding can be resumed via `resume` from within a landingpad or its
    /// successor blocks.
    ///
    /// NOTE: This instruction _must_ be the first instruction in its containing block
    ///
    /// Part of the set of older Itanium C++ exception handling instructions
    pub fn build_landingpad<T: Type>(&self, ty: T) -> LandingPadInst {
        extern "C" {
            fn LLVMBuildLandingPad(
                builder: Builder,
                ty: TypeBase,
                personality_fn: ValueBase,
                hint_num_clauses: u32,
                name: *const std::os::raw::c_char,
            ) -> LandingPadInst;
        }

        unsafe {
            LLVMBuildLandingPad(
                *self.builder,
                ty.base(),
                ValueBase::null(),
                /*hint=*/ 2,
                UNNAMED,
            )
        }
    }

    /// When inserted at the beginning of a block, it marks the block as a landing pad for the unwinder
    /// It acts as the dispatcher across one or more catch handlers, and like invoke, can indicate what
    /// to do if the handler unwinds.
    ///
    /// NOTE: This instruction _must_ be the first instruction in its containing block
    ///
    /// This is part of the set of new exception handling instructions, and are generic across MSVC
    /// structured exception handling and Itanium C++ exceptions. It is a strict superset of the older
    /// instruction set.
    pub fn build_catchswitch(
        &self,
        parent: Option<&Funclet>,
        unwind: Option<Block>,
    ) -> CatchSwitchInst {
        extern "C" {
            fn LLVMBuildCatchSwitch(
                builder: Builder,
                parent: ValueBase,
                unwind: Block,
                hint_num_handlers: u32,
                name: *const std::os::raw::c_char,
            ) -> CatchSwitchInst;
        }

        let unwind = unwind.unwrap_or_else(Block::null);
        let parent = parent.map(|f| f.pad()).unwrap_or_else(ValueBase::null);
        unsafe {
            LLVMBuildCatchSwitch(*self.builder, parent, unwind, /*hint=*/ 5, UNNAMED)
        }
    }

    /// Like `landingpad`, this instruction indicates that its containing block is a catch handler for
    /// a corresponding `catchswitch` instruction. A catchpad will not be visited unless the in-flight
    /// exception matches the given arguments. The arguments correspond to whatever information the
    /// personality routine requires to know if this is an appropriate handler for the exception. In
    /// practice this tends to be a pointer to a global containing a tag string, e.g. `i8** @_ZTIi`.
    ///
    /// Control must exit a catchpad via a `catchret`, it must not resume normal execution, unwind,
    /// or return using `ret`, or the behavior is undefined.
    ///
    /// NOTE: This instruction _must_ be the first instruction in its containing block
    ///
    /// This is part of the set of new exception handling instructions
    pub fn build_catchpad(&self, catchswitch: CatchSwitchInst, args: &[ValueBase]) -> CatchPadInst {
        extern "C" {
            fn LLVMBuildCatchPad(
                builder: Builder,
                parent: ValueBase,
                args: *const ValueBase,
                argc: u32,
                name: *const std::os::raw::c_char,
            ) -> CatchPadInst;
        }

        let argv = args.as_ptr();
        let argc = args.len().try_into().unwrap();
        unsafe { LLVMBuildCatchPad(*self.builder, catchswitch.base(), argv, argc, UNNAMED) }
    }

    /// Similar to `catchpad`, except it is used for the cleanup phase of the unwinder, i.e. it doesn't
    /// resume normal execution, but instead performs some cleanup action and then resumes the unwinder.
    ///
    /// Unlike `catchpad` however, `cleanuppad` doesn't require a `catchswitch` as its parent, as it can
    /// occur in the unwind destination for an invoke directly in cases where the exception isn't handled
    /// but cleanup is required.
    ///
    /// Control must exit a cleanuppad via a `cleanupret`, it must not unwind or return using `ret`, or
    /// the behavior is undefined.
    ///
    /// NOTE: This instruction _must_ be the first instruction in its containing block
    ///
    /// This is part of the set of new exception handling instructions
    pub fn build_cleanuppad(&self, parent: Option<&Funclet>, args: &[ValueBase]) -> CleanupPadInst {
        extern "C" {
            fn LLVMBuildCleanupPad(
                builder: Builder,
                parent: ValueBase,
                args: *const ValueBase,
                len: u32,
                name: *const std::os::raw::c_char,
            ) -> CleanupPadInst;
        }

        let argv = args.as_ptr();
        let argc = args.len().try_into().unwrap();
        let parent = parent.map(|f| f.pad()).unwrap_or_else(ValueBase::null);
        unsafe { LLVMBuildCleanupPad(*self.builder, parent, argv, argc, UNNAMED) }
    }

    /// Resumes normal execution from the body of a `catchpad`
    ///
    /// This instruction ends an in-flight exception whose unwinding was interrupted by its
    /// corresponding `catchpad`. The given `catchpad` reference must be the most recently entered,
    /// not-yet-exited funclet pad, or the behavior is undefined.
    ///
    /// This is part of the set of new exception handling instructions
    pub fn build_catchret(&self, pad: CatchPadInst, dest: Block) -> CatchRetInst {
        extern "C" {
            fn LLVMBuildCatchRet(builder: Builder, pad: ValueBase, block: Block) -> CatchRetInst;
        }

        unsafe { LLVMBuildCatchRet(*self.builder, pad.base(), dest) }
    }

    /// The cleanupret instruction is a terminator with an optional successor
    ///
    /// It requires one argument, which indicates which cleanuppad it exits
    ///
    /// If the optional successor is given, it must be a block beginning with either a cleanuppad
    /// or catchswitch instruction. If it is not given, then unwinding continues in the caller
    ///
    /// NOTE: If the cleanuppad given is not the most recently entered, not-yet-exited funclet pad
    /// (see the EH documentation), the cleanupret's behavior is undefined.
    ///
    /// This is part of the set of new exception handling instructions
    pub fn build_cleanupret(&self, pad: CleanupPadInst, dest: Option<Block>) -> CleanupRetInst {
        extern "C" {
            fn LLVMBuildCleanupRet(
                builder: Builder,
                pad: ValueBase,
                block: Block,
            ) -> CleanupRetInst;
        }

        let dest = dest.unwrap_or_else(Block::null);
        unsafe { LLVMBuildCleanupRet(*self.builder, pad.base(), dest) }
    }

    pub fn build_extractvalue<A: Aggregate>(&self, agg: A, index: usize) -> ExtractValueInst {
        extern "C" {
            fn LLVMBuildExtractValue(
                builder: Builder,
                agg: ValueBase,
                index: u32,
                name: *const std::os::raw::c_char,
            ) -> ExtractValueInst;
        }

        unsafe {
            LLVMBuildExtractValue(
                *self.builder,
                agg.base(),
                index.try_into().unwrap(),
                UNNAMED,
            )
        }
    }

    pub fn build_insertvalue<A: Aggregate, V: Value>(
        &self,
        agg: A,
        element: V,
        index: usize,
    ) -> InsertValueInst {
        extern "C" {
            fn LLVMBuildInsertValue(
                builder: Builder,
                agg: ValueBase,
                element: ValueBase,
                index: u32,
                name: *const std::os::raw::c_char,
            ) -> InsertValueInst;
        }

        unsafe {
            LLVMBuildInsertValue(
                *self.builder,
                agg.base(),
                element.base(),
                index.try_into().unwrap(),
                UNNAMED,
            )
        }
    }

    // TODO: It would be nice if we could make this more type safe and limit the value to one implementing Pointer
    pub fn build_load<V: Value, T: Type>(&self, ty: T, ptr: V) -> LoadInst {
        extern "C" {
            fn LLVMBuildLoad2(
                builder: Builder,
                ty: TypeBase,
                pointer: ValueBase,
                name: *const std::os::raw::c_char,
            ) -> LoadInst;
        }

        unsafe { LLVMBuildLoad2(*self.builder, ty.base(), ptr.base(), UNNAMED) }
    }

    // TODO: It would be nice if we could make this more type safe and limit the pointer value to one implementing Pointer
    pub fn build_store<P: Value, V: Value>(&self, ptr: P, value: V) -> StoreInst {
        extern "C" {
            fn LLVMBuildStore(builder: Builder, value: ValueBase, ptr: ValueBase) -> StoreInst;
        }
        unsafe { LLVMBuildStore(*self.builder, value.base(), ptr.base()) }
    }

    /// Given a pointer to a struct of the given type, produces a value which is a pointer to the `n`th field of the struct
    pub fn build_struct_gep<P: Value>(&self, ty: StructType, ptr: P, n: usize) -> ValueBase {
        extern "C" {
            fn LLVMBuildStructGEP2(
                builder: Builder,
                ty: StructType,
                pointer: ValueBase,
                indices: *const ValueBase,
                num_indices: u32,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        let index_ty = self.get_i32_type();
        let indices = &[ConstantInt::get(index_ty, n as u64, /*sext=*/ false).base()];
        unsafe { LLVMBuildStructGEP2(*self.builder, ty, ptr.base(), indices.as_ptr(), 1, UNNAMED) }
    }

    /// Builds a `getelementptr` instruction that will produce a poison value if any of the following
    /// constraints are violated:
    ///
    /// * The base pointer is an address within the memory bounds of an allocated object
    /// * If the index value must be truncated due to being a larger type, the signed value must be preserved
    /// * Multiplication of the index by the type size does not wrap (in the signed sense, e.g. nsw)
    /// * The successive addition of offsets (without adding the base address) does not wrap
    /// * The successive addition of the current address (interpreted as an unsigned number) and an offset,
    /// interpreted as a signed number, does not wrap the unsigned address space, and remains in bounds of the
    /// allocated object
    /// * In cases where the base is a vector of pointers, these rules apply to each of the computations element-wise
    pub fn build_inbounds_gep<V: Value, T: Type>(
        &self,
        ty: T,
        ptr: V,
        indices: &[usize],
    ) -> ValueBase {
        extern "C" {
            fn LLVMBuildInBoundsGEP2(
                builder: Builder,
                ty: TypeBase,
                ptr: ValueBase,
                indices: *const ValueBase,
                num_indices: u32,
                name: *const std::os::raw::c_char,
            ) -> ValueBase;
        }

        let i32_type = self.get_i32_type();
        let indices_values = indices
            .iter()
            .copied()
            .map(|i| self.build_constant_uint(i32_type, i as u64).base())
            .collect::<Vec<_>>();
        let num_indices = indices_values.len().try_into().unwrap();
        unsafe {
            LLVMBuildInBoundsGEP2(
                *self.builder,
                ty.base(),
                ptr.base(),
                indices_values.as_ptr(),
                num_indices,
                UNNAMED,
            )
        }
    }

    /// Same as `build_inbounds_gep`, but valid in a constant context
    pub fn build_const_inbounds_gep<V: Constant, T: Type>(
        &self,
        ty: T,
        value: V,
        indices: &[usize],
    ) -> ConstantExpr {
        let i32_type = self.get_i32_type();
        let indices_values = indices
            .iter()
            .map(|i| self.build_constant_uint(i32_type, *i as u64).into())
            .collect::<Vec<ConstantValue>>();
        ConstantExpr::inbounds_gep(ty, value, indices_values.as_slice())
    }

    /// Build a global constant with the given type, name, and initializer
    pub fn build_constant<S: Into<StringRef>, C: Constant, T: Type>(
        &self,
        ty: T,
        name: S,
        initializer: C,
    ) -> GlobalVariable {
        let global = self.build_global(ty, name, Some(initializer.base()));
        global.set_constant(true);
        global
    }

    /// Build a global variable with the given type, name, and initializer
    pub fn build_global<S: Into<StringRef>, T: Type>(
        &self,
        ty: T,
        name: S,
        initializer: Option<ValueBase>,
    ) -> GlobalVariable {
        self.module.add_global(ty, name, initializer)
    }

    /// Declare a global, or get the existing declaration if the symbol already exists
    pub fn declare_global<S: Into<StringRef>, T: Type>(&self, name: S, ty: T) -> GlobalVariable {
        self.module.get_or_add_global(ty, name, None)
    }

    /// Defines a global with the given name and type
    ///
    /// If the global already exists, returns None, otherwise returns the newly defined global
    pub fn define_global<S: Into<StringRef>, T: Type>(
        &self,
        name: S,
        ty: T,
    ) -> Option<GlobalVariable> {
        let name = name.into();
        if self.module.get_global(name).is_some() {
            None
        } else {
            Some(self.module.add_global(ty, name, None))
        }
    }

    /// Gets a reference to a global variable with the given name, _if_ it has a definition, otherwise returns None
    pub fn get_defined_value<S: Into<StringRef>>(&self, name: S) -> Option<GlobalVariable> {
        self.module
            .get_global(name)
            .and_then(|gv| if gv.is_declaration() { None } else { Some(gv) })
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

    /// Sets the COMDAT for the given global object to a (ostensibly) unique name, by
    /// using the name associated to the value itself.
    ///
    /// # Rationale
    ///
    /// Externally visible symbols that might appear in multiple codegen units need to appear in
    /// their own comdat section so that the duplicates can be discarded at link time. This can for
    /// example happen for generics when using multiple codegen units. This function simply uses the
    /// value's name as the comdat value to make sure that it is in a 1-to-1 relationship to the
    /// function.
    ///
    /// For more details on COMDAT sections see e.g., http://www.airs.com/blog/archives/52
    pub fn set_unique_comdat<V: GlobalObject>(&self, value: V) {
        value.set_comdat(self.module.get_or_add_comdat(value.name()));
    }

    /// Gets an enum attribute of the given kind, with the default value of 0
    pub fn get_enum_attribute(&self, kind: AttributeKind) -> EnumAttribute {
        EnumAttribute::new(*self.context, kind, None)
    }

    /// Gets a string attribute of the given kind and value
    pub fn get_string_attribute<K, V>(&self, kind: K, value: V) -> StringAttribute
    where
        K: Into<StringRef>,
        V: Into<StringRef>,
    {
        StringAttribute::new(*self.context, kind, value)
    }

    fn apply_default_function_attributes(&self, fun: Function) {
        self.apply_optimization_attributes(fun);
        self.apply_sanitizers(fun);
        // Always annotate functions with the target-cpu they are compiled for.
        // Without this, ThinLTO won't inline Rust functions into Clang generated
        // functions (because Clang annotates functions this way too).
        fun.add_attribute(self.get_string_attribute("target-cpu", self.target_cpu.as_str()));
    }

    fn apply_optimization_attributes(&self, fun: Function) {
        match self.opt_level {
            PassBuilderOptLevel::O0 => {
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::MinSize));
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::OptimizeForSize));
                fun.add_attribute(self.get_enum_attribute(AttributeKind::OptimizeNone));
            }
            PassBuilderOptLevel::Os => {
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::MinSize));
                fun.add_attribute(self.get_enum_attribute(AttributeKind::OptimizeForSize));
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::OptimizeNone));
            }
            PassBuilderOptLevel::Oz => {
                fun.add_attribute(self.get_enum_attribute(AttributeKind::MinSize));
                fun.add_attribute(self.get_enum_attribute(AttributeKind::OptimizeForSize));
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::OptimizeNone));
            }
            _ => {
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::MinSize));
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::OptimizeForSize));
                fun.remove_attribute(self.get_enum_attribute(AttributeKind::OptimizeNone));
            }
        }
    }

    fn apply_sanitizers(&self, fun: Function) {
        if let Some(sanitizer) = self.sanitizer {
            let kind = match sanitizer {
                Sanitizer::Address => AttributeKind::SanitizeAddress,
                Sanitizer::Memory => AttributeKind::SanitizeMemory,
                Sanitizer::Thread => AttributeKind::SanitizeThread,
                Sanitizer::Leak => {
                    return;
                }
            };
            fun.add_attribute(self.get_enum_attribute(kind));
        }
    }

    pub fn get_intrinsic(&self, key: &str) -> Function {
        if let Some(v) = self.intrinsics.borrow().get(key).cloned() {
            return v;
        }

        self.declare_intrinsic(key)
            .unwrap_or_else(|| panic!("unknown intrinsic '{}'", key))
    }

    fn insert_intrinsic(
        &self,
        name: &'static str,
        args: Option<&[TypeBase]>,
        ret: TypeBase,
    ) -> Function {
        let fn_ty = if let Some(args) = args {
            self.get_function_type(ret, args, /* variadic */ false)
        } else {
            self.get_function_type(ret, &[], /* variadic */ true)
        };
        let f = self.declare_function(name, fn_ty);
        f.set_unnamed_address(UnnamedAddr::No);
        self.intrinsics.borrow_mut().insert(name, f);
        f
    }

    fn declare_intrinsic(&self, key: &str) -> Option<Function> {
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
            ($($field_ty:expr),*) => (self.get_struct_type(None, &[$($field_ty),*]).base())
        }

        let t_i8 = self.get_i8_type().base();
        let i8p = self.get_pointer_type(t_i8).base();
        let void = self.get_void_type().base();
        let i1 = self.get_i1_type().base();
        let t_i16 = self.get_i16_type().base();
        let t_i32 = self.get_i32_type().base();
        let t_i64 = self.get_i64_type().base();
        let t_i128 = self.get_i128_type().base();
        let t_f32 = self.get_f32_type().base();
        let t_f64 = self.get_f64_type().base();
        let t_meta = self.get_metadata_type().base();
        let t_token = self.get_token_type().base();

        ifn!("llvm.trap", fn() -> void);
        ifn!("llvm.debugtrap", fn() -> void);
        ifn!("llvm.frameaddress", fn(t_i32) -> i8p);
        ifn!("llvm.sideeffect", fn() -> void);

        ifn!("llvm.powi.f32", fn(t_f32, t_i32) -> t_f32);
        ifn!("llvm.powi.f64", fn(t_f64, t_i32) -> t_f64);

        ifn!("llvm.pow.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.pow.f64", fn(t_f64, t_f64) -> t_f64);

        ifn!("llvm.sqrt.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sqrt.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.sin.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.sin.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.cos.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.cos.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.exp.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.exp2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.exp2.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.log.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.log10.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log10.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.log2.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.log2.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.fma.f32", fn(t_f32, t_f32, t_f32) -> t_f32);
        ifn!("llvm.fma.f64", fn(t_f64, t_f64, t_f64) -> t_f64);

        ifn!("llvm.fabs.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.fabs.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.minnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.minnum.f64", fn(t_f64, t_f64) -> t_f64);
        ifn!("llvm.maxnum.f32", fn(t_f32, t_f32) -> t_f32);
        ifn!("llvm.maxnum.f64", fn(t_f64, t_f64) -> t_f64);

        ifn!("llvm.floor.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.floor.f64", fn(t_f64) -> t_f64);

        ifn!("llvm.ceil.f32", fn(t_f32) -> t_f32);
        ifn!("llvm.ceil.f64", fn(t_f64) -> t_f64);

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

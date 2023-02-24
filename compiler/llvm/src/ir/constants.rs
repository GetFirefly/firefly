use std::mem::MaybeUninit;

use paste::paste;

use super::*;
use crate::support::*;

/// Marker trait for values which are subtypes of llvm::Constant
pub trait Constant: Value {}

/// Represents a reference to a value of any llvm::Constant subclass
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantValue(ValueBase);
impl Value for ConstantValue {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl Constant for ConstantValue {}
impl ConstantValue {
    /// Obtain a constant value referring to the null instance of the given type
    pub fn null<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMConstNull(ty: TypeBase) -> ConstantValue;
        }
        unsafe { LLVMConstNull(ty.base()) }
    }
}
impl TryFrom<ValueBase> for ConstantValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::BlockAddress
            | ValueKind::ConstantExpr
            | ValueKind::ConstantArray
            | ValueKind::ConstantStruct
            | ValueKind::ConstantVector
            | ValueKind::Undef
            | ValueKind::Poison
            | ValueKind::ConstantAggregateZero
            | ValueKind::ConstantDataArray
            | ValueKind::ConstantDataVector
            | ValueKind::ConstantInt
            | ValueKind::ConstantFP
            | ValueKind::ConstantPointerNull
            | ValueKind::ConstantTokenNone => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl From<Function> for ConstantValue {
    fn from(fun: Function) -> Self {
        Self(fun.base())
    }
}
impl TryFrom<GlobalVariable> for ConstantValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: GlobalVariable) -> Result<Self, Self::Error> {
        if value.is_constant() {
            Ok(Self(value.base()))
        } else {
            value.base().try_into()
        }
    }
}

/// Represents an undefined initial value of some type
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct UndefValue(ValueBase);
impl Value for UndefValue {
    fn is_undef(&self) -> bool {
        true
    }

    #[inline(always)]
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl UndefValue {
    pub fn get<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMGetUndef(ty: TypeBase) -> ValueBase;
        }
        Self(unsafe { LLVMGetUndef(ty.base()) })
    }
}
impl TryFrom<ValueBase> for UndefValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        if value.is_undef() {
            Ok(Self(value))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Into<ConstantValue> for UndefValue {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl Into<ValueBase> for UndefValue {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Represents the poison value
///
/// All uses of poison values are themselves poisoned as a result.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct PoisonValue(ValueBase);
impl Value for PoisonValue {
    fn is_poison(&self) -> bool {
        true
    }

    #[inline(always)]
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl PoisonValue {
    pub fn get<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMGetPoison(ty: TypeBase) -> ValueBase;
        }
        Self(unsafe { LLVMGetPoison(ty.base()) })
    }
}
impl TryFrom<ValueBase> for PoisonValue {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        if value.is_poison() {
            Ok(Self(value))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Into<ConstantValue> for PoisonValue {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl Into<ValueBase> for PoisonValue {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Represents a constant integer literal
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantInt(ValueBase);
impl Constant for ConstantInt {}
impl Value for ConstantInt {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantInt {
    /// Obtain a constant value for an integer type.
    ///
    /// The returned value corresponds to `llvm::ConstantInt`
    ///
    /// * `ty` should be the integer type of the resulting value
    /// * `value` should be the desired integer value as a u64
    /// * `sext` should be set to true if the integer should be sign-extended to fit the type
    ///
    /// NOTE: For signed integers, cast them to u64 and then set `sext` to true
    pub fn get(ty: IntegerType, value: u64, sext: bool) -> Self {
        extern "C" {
            fn LLVMConstInt(ty: IntegerType, value: u64, sext: bool) -> ConstantInt;
        }
        unsafe { LLVMConstInt(ty, value, sext) }
    }

    /// Obtain a constant value referring to the instance of a type consisting of all ones
    pub fn all_ones(ty: IntegerType) -> Self {
        extern "C" {
            fn LLVMConstAllOnes(ty: IntegerType) -> ConstantInt;
        }
        unsafe { LLVMConstAllOnes(ty) }
    }

    /// Obtain the sign-extended value of this constant
    pub fn value(self) -> i64 {
        extern "C" {
            fn LLVMConstIntGetSExtValue(value: ConstantInt) -> i64;
        }
        unsafe { LLVMConstIntGetSExtValue(self) }
    }

    /// Obtain the zero-extended value of this constant
    pub fn value_unsigned(self) -> u64 {
        extern "C" {
            fn LLVMConstIntGetZExtValue(value: ConstantInt) -> u64;
        }
        unsafe { LLVMConstIntGetZExtValue(self) }
    }
}
impl Into<ConstantValue> for ConstantInt {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantInt {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantInt => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents a constant floating-pointer literal
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantFloat(ValueBase);
impl Constant for ConstantFloat {}
impl Value for ConstantFloat {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantFloat {
    /// Obtain a constant value referring to a double floating point value.
    ///
    /// * `ty` should be the float type of the resulting value
    /// * `value` should be the desired value as a 64-bit floating point value
    pub fn get(ty: FloatType, value: f64) -> Self {
        extern "C" {
            fn LLVMConstReal(ty: FloatType, value: f64) -> ConstantFloat;
        }
        unsafe { LLVMConstReal(ty, value) }
    }

    /// Obtain the value of this floating-point constant
    ///
    /// This conversion can potentially be lossy depending on the underlying type.
    pub fn value(self) -> f64 {
        extern "C" {
            fn LLVMConstRealGetDouble(value: ConstantFloat, lossy: *mut bool) -> f64;
        }
        let mut lossy = MaybeUninit::uninit();
        unsafe { LLVMConstRealGetDouble(self, lossy.as_mut_ptr()) }
    }
}
impl Into<ConstantValue> for ConstantFloat {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantFloat {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantFP => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents a constant null pointer
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantPointerNull(ValueBase);
impl Constant for ConstantPointerNull {}
impl Pointer for ConstantPointerNull {}
impl Value for ConstantPointerNull {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantPointerNull {
    /// Obtain a constant that is a constant pointer to NULL for the given type
    pub fn get<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMConstPointerNull(ty: TypeBase) -> ConstantPointerNull;
        }
        unsafe { LLVMConstPointerNull(ty.base()) }
    }
}
impl Into<ConstantValue> for ConstantPointerNull {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantPointerNull {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantPointerNull => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents a constant null-terminated string literal
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantString(ValueBase);
impl Constant for ConstantString {}
impl Value for ConstantString {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantString {
    /// Obtain the value of this string constant
    pub fn get(self) -> StringRef {
        extern "C" {
            fn LLVMGetAsString(value: ConstantString, len: *mut usize) -> *const u8;
        }
        unsafe {
            let mut len = MaybeUninit::uninit();
            let data = LLVMGetAsString(self, len.as_mut_ptr());
            StringRef {
                data,
                len: len.assume_init(),
            }
        }
    }
}
impl Into<ConstantValue> for ConstantString {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantString {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantDataArray => {
                if value.is_constant_string() {
                    Ok(Self(value))
                } else {
                    Err(InvalidTypeCastError)
                }
            }
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Marker trait for constant values of aggregate type (struct, array, vectors)
pub trait ConstantAggregate: Constant + Aggregate {}
impl<T: Constant + Aggregate> ConstantAggregate for T {}

/// Represents a constant struct value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantStruct(ValueBase);
impl Constant for ConstantStruct {}
impl Aggregate for ConstantStruct {}
impl Value for ConstantStruct {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantStruct {
    pub fn get_named(ty: StructType, values: &[ConstantValue]) -> Self {
        extern "C" {
            fn LLVMConstNamedStruct(
                ty: StructType,
                values: *const ConstantValue,
                len: u32,
            ) -> ConstantStruct;
        }
        unsafe { LLVMConstNamedStruct(ty, values.as_ptr(), values.len().try_into().unwrap()) }
    }
}
impl Into<ConstantValue> for ConstantStruct {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantStruct {
    type Error = InvalidTypeCastError;
    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantStruct => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl Into<ValueBase> for ConstantStruct {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Represents a constant array value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantArray(ValueBase);
impl Constant for ConstantArray {}
impl Aggregate for ConstantArray {}
impl Value for ConstantArray {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantArray {
    pub fn get<T: Type>(ty: T, values: &[ConstantValue]) -> Self {
        extern "C" {
            fn LLVMConstArray(
                element_ty: TypeBase,
                values: *const ConstantValue,
                len: u32,
            ) -> ConstantArray;
        }
        unsafe { LLVMConstArray(ty.base(), values.as_ptr(), values.len().try_into().unwrap()) }
    }
}
impl Into<ConstantValue> for ConstantArray {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl TryFrom<ValueBase> for ConstantArray {
    type Error = InvalidTypeCastError;
    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantArray => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl Into<ValueBase> for ConstantArray {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Represents a constant vector value
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct ConstantVector(ValueBase);
impl Constant for ConstantVector {}
impl Value for ConstantVector {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantVector {
    pub fn get(values: &[ConstantValue]) -> Self {
        extern "C" {
            fn LLVMConstVector(values: *const ConstantValue, len: u32) -> ConstantVector;
        }
        unsafe { LLVMConstVector(values.as_ptr(), values.len().try_into().unwrap()) }
    }
}
impl TryFrom<ValueBase> for ConstantVector {
    type Error = InvalidTypeCastError;
    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantVector => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl Into<ConstantValue> for ConstantVector {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl Into<ValueBase> for ConstantVector {
    fn into(self) -> ValueBase {
        self.0
    }
}

macro_rules! const_unary_op {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            const_unary_op!($name, $mnemonic, [<LLVMConst $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $extern:ident) => {
        pub fn $mnemonic<C: Constant>(value: C) -> Self {
            extern "C" {
                fn $extern(value: ValueBase) -> ConstantExpr;
            }
            unsafe { $extern(value.base()) }
        }
    };
}

macro_rules! const_cast_op {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            const_cast_op!($name, $mnemonic, [<LLVMConst $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $extern:ident) => {
        pub fn $mnemonic<C: Constant, T: Type>(value: C, ty: T) -> Self {
            extern "C" {
                fn $extern(value: ValueBase, to_type: TypeBase) -> ConstantExpr;
            }
            unsafe { $extern(value.base(), ty.base()) }
        }
    };
}

macro_rules! const_binary_op {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            const_binary_op!($name, $mnemonic, [<LLVMConst $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $extern:ident) => {
        pub fn $mnemonic<L, R>(lhs: L, rhs: R) -> Self
        where
            L: Constant,
            R: Constant,
        {
            extern "C" {
                fn $extern(lhs: ValueBase, rhs: ValueBase) -> ConstantExpr;
            }
            unsafe { $extern(lhs.base(), rhs.base()) }
        }
    };
}

/// Represents all forms of constant expressions
#[repr(transparent)]
pub struct ConstantExpr(ValueBase);
impl Constant for ConstantExpr {}
impl Value for ConstantExpr {
    fn base(&self) -> ValueBase {
        self.0
    }
}
impl ConstantExpr {
    pub fn opcode(self) -> Opcode {
        extern "C" {
            fn LLVMGetConstOpcode(value: ConstantExpr) -> Opcode;
        }
        unsafe { LLVMGetConstOpcode(self) }
    }

    pub fn align_of<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMAlignOf(ty: TypeBase) -> ConstantExpr;
        }
        unsafe { LLVMAlignOf(ty.base()) }
    }

    pub fn size_of<T: Type>(ty: T) -> Self {
        extern "C" {
            fn LLVMSizeOf(ty: TypeBase) -> ConstantExpr;
        }
        unsafe { LLVMSizeOf(ty.base()) }
    }

    const_unary_op!(Neg, neg);
    const_unary_op!(NSWNeg, nsw_neg);
    const_unary_op!(NUWNeg, nuw_neg);
    const_unary_op!(FNeg, fneg);
    const_unary_op!(Not, not);

    const_binary_op!(Add, add);
    const_binary_op!(NSWAdd, nsw_add);
    const_binary_op!(NUWAdd, nuw_add);
    const_binary_op!(FAdd, fadd);
    const_binary_op!(Sub, sub);
    const_binary_op!(NSWSub, nsw_sub);
    const_binary_op!(NUWSub, nuw_sub);
    const_binary_op!(FSub, fsub);
    const_binary_op!(Mul, mul);
    const_binary_op!(NSWMul, nsw_mul);
    const_binary_op!(NUWMul, nuw_mul);
    const_binary_op!(FMul, fmul);
    const_binary_op!(UDiv, udiv);
    const_binary_op!(ExactUDiv, exact_udiv);
    const_binary_op!(SDiv, sdiv);
    const_binary_op!(ExactSDiv, exact_sdiv);
    const_binary_op!(FDiv, fdiv);
    const_binary_op!(URem, urem);
    const_binary_op!(SRem, srem);
    const_binary_op!(FRem, frem);
    const_binary_op!(And, and);
    const_binary_op!(Or, or);
    const_binary_op!(Xor, xor);
    const_binary_op!(Shl, shl);
    const_binary_op!(LShr, lhsr);
    const_binary_op!(Ashr, ashr);

    const_cast_op!(Trunc, trunc);
    const_cast_op!(SExt, sext);
    const_cast_op!(ZExt, zext);
    const_cast_op!(FPTrunc, fp_trunc);
    const_cast_op!(FPExt, fp_ext);
    const_cast_op!(UIToFP, unsigned_to_float);
    const_cast_op!(SIToFP, signed_to_float);
    const_cast_op!(FPToUI, float_to_unsigned);
    const_cast_op!(FPToSI, float_to_signed);
    const_cast_op!(PtrToInt, ptr_to_int);
    const_cast_op!(IntToPtr, int_to_ptr);
    const_cast_op!(BitCast, bitcast);
    const_cast_op!(AddrSpaceCast, addrspace_cast);
    const_cast_op!(ZExtOrBitCast, zext_or_bitcast);
    const_cast_op!(SExtOrBitCast, sext_or_bitcast);
    const_cast_op!(TruncOrBitCast, trunc_or_bitcast);
    const_cast_op!(PointerCast, pointer_cast);
    const_cast_op!(IntCast, int_cast);
    const_cast_op!(FPCast, fp_cast);

    pub fn icmp<L, R>(predicate: ICmp, lhs: ConstantValue, rhs: ConstantValue) -> Self
    where
        L: Constant,
        R: Constant,
    {
        extern "C" {
            fn LLVMConstICmp(
                predicate: ICmp,
                lhs: ConstantValue,
                rhs: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstICmp(predicate, lhs, rhs) }
    }

    pub fn fcmp<L, R>(predicate: FCmp, lhs: ConstantValue, rhs: ConstantValue) -> Self
    where
        L: Constant,
        R: Constant,
    {
        extern "C" {
            fn LLVMConstFCmp(
                predicate: FCmp,
                lhs: ConstantValue,
                rhs: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstFCmp(predicate, lhs, rhs) }
    }

    pub fn gep<C: Constant, T: Type>(ty: T, constant: C, indices: &[ConstantValue]) -> Self {
        extern "C" {
            fn LLVMConstGEP2(
                ty: TypeBase,
                constant: ValueBase,
                indices: *const ConstantValue,
                len: u32,
            ) -> ConstantExpr;
        }
        unsafe {
            LLVMConstGEP2(
                ty.base(),
                constant.base(),
                indices.as_ptr(),
                indices.len().try_into().unwrap(),
            )
        }
    }

    pub fn inbounds_gep<C: Constant, T: Type>(
        ty: T,
        constant: C,
        indices: &[ConstantValue],
    ) -> Self {
        extern "C" {
            fn LLVMConstInBoundsGEP2(
                ty: TypeBase,
                constant: ValueBase,
                indices: *const ConstantValue,
                len: u32,
            ) -> ConstantExpr;
        }
        unsafe {
            LLVMConstInBoundsGEP2(
                ty.base(),
                constant.base(),
                indices.as_ptr(),
                indices.len().try_into().unwrap(),
            )
        }
    }

    pub fn select(cond: ConstantValue, if_true: ConstantValue, if_false: ConstantValue) -> Self {
        extern "C" {
            fn LLVMConstSelect(
                cond: ConstantValue,
                if_true: ConstantValue,
                if_false: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstSelect(cond, if_true, if_false) }
    }

    pub fn extract_element(vector: ConstantVector, index: ConstantValue) -> Self {
        extern "C" {
            fn LLVMConstExtractElement(
                vector: ConstantVector,
                index: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstExtractElement(vector, index) }
    }

    pub fn insert_element(
        vector: ConstantVector,
        value: ConstantValue,
        index: ConstantValue,
    ) -> Self {
        extern "C" {
            fn LLVMConstInsertElement(
                vector: ConstantVector,
                value: ConstantValue,
                index: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstInsertElement(vector, value, index) }
    }

    pub fn shuffle_vector(a: ConstantVector, b: ConstantVector, mask: ConstantValue) -> Self {
        extern "C" {
            fn LLVMConstShuffleVector(
                a: ConstantVector,
                b: ConstantVector,
                mask: ConstantValue,
            ) -> ConstantExpr;
        }
        unsafe { LLVMConstShuffleVector(a, b, mask) }
    }

    pub fn extract_value<A: ConstantAggregate>(aggregate: A, indices: &[ConstantValue]) -> Self {
        extern "C" {
            fn LLVMConstExtractValue(
                aggregate: ValueBase,
                indices: *const ConstantValue,
                num_indices: u32,
            ) -> ConstantExpr;
        }
        unsafe {
            LLVMConstExtractValue(
                aggregate.base(),
                indices.as_ptr(),
                indices.len().try_into().unwrap(),
            )
        }
    }

    pub fn insert_value<A: ConstantAggregate>(
        aggregate: A,
        value: ConstantValue,
        indices: &[ConstantValue],
    ) -> Self {
        extern "C" {
            fn LLVMConstInsertValue(
                aggregate: ValueBase,
                value: ConstantValue,
                indices: *const ConstantValue,
                num_indices: u32,
            ) -> ConstantExpr;
        }
        unsafe {
            LLVMConstInsertValue(
                aggregate.base(),
                value,
                indices.as_ptr(),
                indices.len().try_into().unwrap(),
            )
        }
    }

    pub fn block_address(function: Function, block: Block) -> Self {
        extern "C" {
            fn LLVMBlockAddress(function: Function, block: Block) -> ConstantExpr;
        }
        unsafe { LLVMBlockAddress(function, block) }
    }
}
impl TryFrom<ValueBase> for ConstantExpr {
    type Error = InvalidTypeCastError;
    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::ConstantExpr => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}
impl Into<ConstantValue> for ConstantExpr {
    #[inline]
    fn into(self) -> ConstantValue {
        ConstantValue(self.0)
    }
}
impl Into<ValueBase> for ConstantExpr {
    fn into(self) -> ValueBase {
        self.0
    }
}

/// Represents what dialect a string of inline assembly is expressed in
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum InlineAsmDialect {
    ATT = 0,
    Intel,
}
impl Default for InlineAsmDialect {
    fn default() -> Self {
        Self::ATT
    }
}

bitflags::bitflags! {
    /// Represents the various options one can pass when creating an inline assembly expression
    pub struct InlineAsmFlags: u32 {
        /// By default, no flags are set, and as such, the following is
        /// assumed about the inline assembly expression:
        ///
        /// * it is expressed in ATT dialect
        /// * it has no side effects not expressed in the constraints
        /// * it does not require stack alignment
        /// * it cannot unwind
        const DEFAULT = 0;

        /// When the inline assembly has side effects which are not in the constraints list,
        /// the expression must be flagged with HAS_SIDE_EFFECTS
        const HAS_SIDE_EFFECTS = 1 << 0;
        /// If the inline assembly requires the stack to be aligned, this flag should be
        /// set, which will result in the compiler generating the usual stack alignment code
        /// in the function prologue
        const ALIGNSTACK = 1 << 1;
        /// If the inline assembly might unwind the stack, this flag must be set, otherwise
        /// unwinding metadata will not be emitted
        const CAN_UNWIND = 1 << 2;
        /// By default inline assembly is expressed in ATT dialect, but you can set this flag
        /// if you prefer to write in Intel syntax.
        const INTEL_DIALECT = 1 << 3;
    }
}
impl Default for InlineAsmFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl ConstantExpr {
    /// Generates an inline assembly expression that has the given type, constraints, and flags
    pub fn inline_asm<T, A, C>(ty: T, asm: A, constraints: C, flags: InlineAsmFlags) -> Self
    where
        T: Type,
        A: Into<StringRef>,
        C: Into<StringRef>,
    {
        extern "C" {
            fn LLVMGetInlineAsm(
                ty: TypeBase,
                asm: *const u8,
                asm_len: usize,
                constraints: *const u8,
                constraints_len: usize,
                has_side_effects: bool,
                is_align_stack: bool,
                dialect: InlineAsmDialect,
                can_unwind: bool,
            ) -> ConstantExpr;
        }
        let asm = asm.into();
        let constraints = constraints.into();
        let has_side_effects = flags.contains(InlineAsmFlags::HAS_SIDE_EFFECTS);
        let alignstack = flags.contains(InlineAsmFlags::ALIGNSTACK);
        let can_unwind = flags.contains(InlineAsmFlags::CAN_UNWIND);
        let dialect = if flags.contains(InlineAsmFlags::INTEL_DIALECT) {
            InlineAsmDialect::Intel
        } else {
            InlineAsmDialect::ATT
        };
        unsafe {
            LLVMGetInlineAsm(
                ty.base(),
                asm.data,
                asm.len,
                constraints.data,
                constraints.len,
                has_side_effects,
                alignstack,
                dialect,
                can_unwind,
            )
        }
    }
}

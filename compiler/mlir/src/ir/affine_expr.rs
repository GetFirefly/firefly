use std::ffi::c_void;
use std::fmt::{self, Display};
use std::ops::{Add, Div, Mul};

use paste::paste;

use super::*;
use crate::support::{self, MlirStringCallback};
use crate::Context;

extern "C" {
    type MlirAffineExpr;
}

/// This trait is implemented by all affine expression types:
///
/// * dimensions
/// * symbols
/// * constants
/// * add
/// * mul
/// * mod
/// * ceildiv
/// * floordiv
pub trait AffineExpr {
    /// Returns the context in which this expression was created
    fn context(&self) -> Context;
    /// Prints the textual representation of this expression to stderr
    fn dump(&self);
    /// Returns this expression as an AffineExprBase
    fn base(&self) -> AffineExprBase;
    /// Checks whether the given affine expression is made out of only symbols and
    /// constants.
    fn is_symbolic_or_constant(&self) -> bool;
    /// Checks whether the given affine expression is a pure affine expression, i.e.
    /// mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
    fn is_pure_affine(&self) -> bool;
    /// Returns the greatest known integral divisor of this affine expression. The
    /// result is always positive.
    fn largest_known_divisor(&self) -> i64;
    /// Checks whether the given affine expression is a multiple of 'factor'.
    fn is_multiple_of(&self, factor: i64) -> bool;
    /// Checks whether the given affine expression involves AffineDimExpr
    /// 'position'.
    fn is_function_of_dim(&self, position: usize) -> bool;
    /// Composes the given map with the given expression.
    fn compose(&self, map: AffineMap) -> AffineExprBase;
}
impl<T: AsRef<AffineExprBase>> AffineExpr for T {
    #[inline(always)]
    fn context(&self) -> Context {
        self.as_ref().context()
    }
    #[inline(always)]
    fn dump(&self) {
        self.as_ref().dump()
    }
    #[inline(always)]
    fn base(&self) -> AffineExprBase {
        *self.as_ref()
    }
    fn is_symbolic_or_constant(&self) -> bool {
        self.as_ref().is_symbolic_or_constant()
    }
    fn is_pure_affine(&self) -> bool {
        self.as_ref().is_pure_affine()
    }
    fn largest_known_divisor(&self) -> i64 {
        self.as_ref().largest_known_divisor()
    }
    fn is_multiple_of(&self, factor: i64) -> bool {
        self.as_ref().is_multiple_of(factor)
    }
    fn is_function_of_dim(&self, position: usize) -> bool {
        self.as_ref().is_function_of_dim(position)
    }
    fn compose(&self, map: AffineMap) -> AffineExprBase {
        self.as_ref().compose(map)
    }
}

/// Represents an affine dimension expression
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineDimExpr(AffineExprBase);
impl AffineDimExpr {
    /// Creates an affine dimension expression with 'position'.
    pub fn get(context: Context, position: usize) -> Self {
        unsafe { mlir_affine_dim_expr_get(context, position) }
    }

    /// Returns the position of the given affine dimension expression.
    pub fn position(self) -> usize {
        unsafe { mlir_affine_dim_expr_get_position(self) }
    }
}
impl AsRef<AffineExprBase> for AffineDimExpr {
    #[inline]
    fn as_ref(&self) -> &AffineExprBase {
        &self.0
    }
}
impl Into<AffineExprBase> for AffineDimExpr {
    #[inline]
    fn into(self) -> AffineExprBase {
        self.0
    }
}
impl TryFrom<AffineExprBase> for AffineDimExpr {
    type Error = InvalidTypeCastError;

    fn try_from(expr: AffineExprBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_affine_expr_isa_dim(expr) } {
            Ok(Self(expr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for AffineDimExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for AffineDimExpr {}
impl PartialEq for AffineDimExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AffineExprBase> for AffineDimExpr {
    fn eq(&self, other: &AffineExprBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAffineExprIsADim"]
    fn mlir_affine_expr_isa_dim(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineDimExprGet"]
    fn mlir_affine_dim_expr_get(context: Context, position: usize) -> AffineDimExpr;
    #[link_name = "mlirAffineDimExprGetPosition"]
    fn mlir_affine_dim_expr_get_position(expr: AffineDimExpr) -> usize;
}

/// Represents an affine symbol expression
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineSymbolExpr(AffineExprBase);
impl AffineSymbolExpr {
    /// Creates an affine symbol expression with 'position'.
    pub fn get(context: Context, position: usize) -> Self {
        unsafe { mlir_affine_symbol_expr_get(context, position) }
    }

    /// Returns the position of the given affine symbol expression.
    pub fn position(self) -> usize {
        unsafe { mlir_affine_symbol_expr_get_position(self) }
    }
}
impl AsRef<AffineExprBase> for AffineSymbolExpr {
    #[inline]
    fn as_ref(&self) -> &AffineExprBase {
        &self.0
    }
}
impl Into<AffineExprBase> for AffineSymbolExpr {
    #[inline]
    fn into(self) -> AffineExprBase {
        self.0
    }
}
impl TryFrom<AffineExprBase> for AffineSymbolExpr {
    type Error = InvalidTypeCastError;

    fn try_from(expr: AffineExprBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_affine_expr_isa_symbol(expr) } {
            Ok(Self(expr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for AffineSymbolExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for AffineSymbolExpr {}
impl PartialEq for AffineSymbolExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AffineExprBase> for AffineSymbolExpr {
    fn eq(&self, other: &AffineExprBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAffineExprIsASymbol"]
    fn mlir_affine_expr_isa_symbol(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineSymbolExprGet"]
    fn mlir_affine_symbol_expr_get(context: Context, position: usize) -> AffineSymbolExpr;
    #[link_name = "mlirAffineSymbolExprGetPosition"]
    fn mlir_affine_symbol_expr_get_position(expr: AffineSymbolExpr) -> usize;
}

/// Represents an affine constant expression
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineConstantExpr(AffineExprBase);
impl AffineConstantExpr {
    /// Creates an affine constant expression with 'value'.
    pub fn get(context: Context, value: i64) -> Self {
        unsafe { mlir_affine_constant_expr_get(context, value) }
    }

    /// Returns the value of the given affine constant expression.
    pub fn value(self) -> i64 {
        unsafe { mlir_affine_constant_expr_get_value(self) }
    }
}
impl AsRef<AffineExprBase> for AffineConstantExpr {
    #[inline]
    fn as_ref(&self) -> &AffineExprBase {
        &self.0
    }
}
impl Into<AffineExprBase> for AffineConstantExpr {
    #[inline]
    fn into(self) -> AffineExprBase {
        self.0
    }
}
impl TryFrom<AffineExprBase> for AffineConstantExpr {
    type Error = InvalidTypeCastError;

    fn try_from(expr: AffineExprBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_affine_expr_isa_constant(expr) } {
            Ok(Self(expr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}
impl Display for AffineConstantExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", &self.0)
    }
}
impl Eq for AffineConstantExpr {}
impl PartialEq for AffineConstantExpr {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl PartialEq<AffineExprBase> for AffineConstantExpr {
    fn eq(&self, other: &AffineExprBase) -> bool {
        self.0.eq(other)
    }
}

extern "C" {
    #[link_name = "mlirAffineExprIsAConstant"]
    fn mlir_affine_expr_isa_constant(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineConstantExprGet"]
    fn mlir_affine_constant_expr_get(context: Context, value: i64) -> AffineConstantExpr;
    #[link_name = "mlirAffineConstantExprGetValue"]
    fn mlir_affine_constant_expr_get_value(expr: AffineConstantExpr) -> i64;
}

/// This trait is implemented by all affine expression types which are binary operators
pub trait BinaryAffineExpr: AffineExpr {
    /// Get the left-hand operand of the expression
    fn left(&self) -> AffineExprBase {
        unsafe { mlir_affine_binary_op_expr_get_lhs(self.base()) }
    }

    /// Get the left-hand operand of the expression
    fn right(&self) -> AffineExprBase {
        unsafe { mlir_affine_binary_op_expr_get_rhs(self.base()) }
    }
}

/// This struct is used to promote an AffineExprBase to a form
/// that implements `BinaryAffineExpr` without needing to care
/// about the specific operator
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineExprBinaryOp(AffineExprBase);
impl BinaryAffineExpr for AffineExprBinaryOp {}
impl AsRef<AffineExprBase> for AffineExprBinaryOp {
    #[inline]
    fn as_ref(&self) -> &AffineExprBase {
        &self.0
    }
}
impl Into<AffineExprBase> for AffineExprBinaryOp {
    #[inline]
    fn into(self) -> AffineExprBase {
        self.0
    }
}
impl<T: BinaryAffineExpr> From<&T> for AffineExprBinaryOp {
    #[inline]
    fn from(expr: &T) -> Self {
        Self(expr.base())
    }
}
impl TryFrom<AffineExprBase> for AffineExprBinaryOp {
    type Error = InvalidTypeCastError;

    fn try_from(expr: AffineExprBase) -> Result<Self, Self::Error> {
        if unsafe { mlir_affine_expr_isa_binary(expr) } {
            Ok(Self(expr))
        } else {
            Err(InvalidTypeCastError)
        }
    }
}

extern "C" {
    #[link_name = "mlirAffineExprIsABinary"]
    fn mlir_affine_expr_isa_binary(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineBinaryOpExprGetLHS"]
    fn mlir_affine_binary_op_expr_get_lhs(expr: AffineExprBase) -> AffineExprBase;
    #[link_name = "mlirAffineBinaryOpExprGetRHS"]
    fn mlir_affine_binary_op_expr_get_rhs(expr: AffineExprBase) -> AffineExprBase;
}

macro_rules! affine_expr_binary_op {
    ($name:ident, $mnemonic:ident) => {
        paste! {
            affine_expr_binary_op!($name, $mnemonic, [<Affine $name Expr>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident) => {
        /// Represents an affine $mnemonic expression
        #[repr(transparent)]
        #[derive(Copy, Clone)]
        pub struct $ty(AffineExprBase);
        impl BinaryAffineExpr for $ty {}
        impl AsRef<AffineExprBase> for $ty {
            #[inline]
            fn as_ref(&self) -> &AffineExprBase {
                &self.0
            }
        }
        impl $ty {
            pub fn get<A, B>(a: A, b: B) -> Self
            where
                A: AffineExpr,
                B: AffineExpr,
            {
                paste! {
                    unsafe { [<mlir_affine_ $mnemonic _expr_get>](a.base(), b.base()) }
                }
            }
        }
        impl Into<AffineExprBase> for $ty {
            fn into(self) -> AffineExprBase {
                self.0
            }
        }
        impl TryFrom<AffineExprBase> for $ty {
            type Error = InvalidTypeCastError;

            fn try_from(expr: AffineExprBase) -> Result<Self, Self::Error> {
                let truth = paste! {
                    unsafe { [<mlir_affine_expr_isa_ $mnemonic>](expr) }
                };
                if truth {
                    Ok(Self(expr))
                } else {
                    Err(InvalidTypeCastError)
                }
            }
        }
        impl std::fmt::Display for $ty {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{}", &self.0)
            }
        }
        impl Eq for $ty {}
        impl PartialEq for $ty {
            fn eq(&self, other: &Self) -> bool {
                self.0 == other.0
            }
        }
        impl PartialEq<AffineExprBase> for $ty {
            fn eq(&self, other: &AffineExprBase) -> bool {
                self.0.eq(other)
            }
        }

        paste! {
            affine_expr_binary_op!($name, $mnemonic, $ty, [<mlirAffine $ty Get>], [<mlirAffineExprIsA $name>]);
        }
    };

    ($name:ident, $mnemonic:ident, $ty:ident, $get_name:ident, $isa_name:ident) => {
        extern "C" {
            paste! {
                #[link_name = stringify!($get_name)]
                fn [<mlir_affine_ $mnemonic _expr_get>](x: AffineExprBase, y: AffineExprBase) -> $ty;
                #[link_name = stringify!($isa_name)]
                fn [<mlir_affine_expr_isa_ $mnemonic>](expr: AffineExprBase) -> bool;
            }
        }
    };
}

affine_expr_binary_op!(Add, add);
affine_expr_binary_op!(Mul, mul);
affine_expr_binary_op!(Mod, mod);
affine_expr_binary_op!(FloorDiv, floor_div);
affine_expr_binary_op!(CeilDiv, ceil_div);

macro_rules! impl_binary_op {
    ($op:ident, $mnemonic:ident, $ty:ident) => {
        impl_binary_op!($op, $mnemonic, $mnemonic, $ty);
    };

    ($op:ident, $op_lower:ident, $mnemonic:ident, $ty:ident) => {
        impl<A: AffineExpr> $op<A> for $ty {
            type Output = AffineExprBase;

            #[inline]
            fn $op_lower(self, rhs: A) -> Self::Output {
                paste! {
                    unsafe { [<mlir_affine_ $mnemonic _expr_get>](self.base(), rhs.base()) }.base()
                }
            }
        }
    };
}

impl_binary_op!(Add, add, AffineExprBase);
impl_binary_op!(Mul, mul, AffineExprBase);
impl_binary_op!(Div, div, floor_div, AffineExprBase);

impl_binary_op!(Add, add, AffineDimExpr);
impl_binary_op!(Mul, mul, AffineDimExpr);
impl_binary_op!(Div, div, floor_div, AffineDimExpr);

impl_binary_op!(Add, add, AffineSymbolExpr);
impl_binary_op!(Mul, mul, AffineSymbolExpr);
impl_binary_op!(Div, div, floor_div, AffineSymbolExpr);

impl_binary_op!(Add, add, AffineConstantExpr);
impl_binary_op!(Mul, mul, AffineConstantExpr);
impl_binary_op!(Div, div, floor_div, AffineConstantExpr);

impl_binary_op!(Add, add, AffineExprBinaryOp);
impl_binary_op!(Mul, mul, AffineExprBinaryOp);
impl_binary_op!(Div, div, floor_div, AffineExprBinaryOp);

impl_binary_op!(Add, add, AffineAddExpr);
impl_binary_op!(Mul, mul, AffineAddExpr);
impl_binary_op!(Div, div, floor_div, AffineAddExpr);

impl_binary_op!(Add, add, AffineMulExpr);
impl_binary_op!(Mul, mul, AffineMulExpr);
impl_binary_op!(Div, div, floor_div, AffineMulExpr);

impl_binary_op!(Add, add, AffineFloorDivExpr);
impl_binary_op!(Mul, mul, AffineFloorDivExpr);
impl_binary_op!(Div, div, floor_div, AffineFloorDivExpr);

impl_binary_op!(Add, add, AffineCeilDivExpr);
impl_binary_op!(Mul, mul, AffineCeilDivExpr);
impl_binary_op!(Div, div, floor_div, AffineCeilDivExpr);

/// Represents the base type of all affine expressions
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineExprBase(*mut MlirAffineExpr);
impl AffineExpr for AffineExprBase {
    fn context(&self) -> Context {
        unsafe { mlir_affine_expr_get_context(*self) }
    }

    fn dump(&self) {
        unsafe {
            mlir_affine_expr_dump(*self);
        }
    }

    #[inline(always)]
    fn base(&self) -> AffineExprBase {
        *self
    }

    /// Checks whether the given affine expression is made out of only symbols and
    /// constants.
    fn is_symbolic_or_constant(&self) -> bool {
        unsafe { mlir_affine_expr_is_symbolic_or_constant(*self) }
    }
    /// Checks whether the given affine expression is a pure affine expression, i.e.
    /// mul, floordiv, ceildic, and mod is only allowed w.r.t constants.
    fn is_pure_affine(&self) -> bool {
        unsafe { mlir_affine_expr_is_pure_affine(*self) }
    }
    /// Returns the greatest known integral divisor of this affine expression. The
    /// result is always positive.
    fn largest_known_divisor(&self) -> i64 {
        unsafe { mlir_affine_expr_get_largest_known_divisor(*self) }
    }
    /// Checks whether the given affine expression is a multiple of 'factor'.
    fn is_multiple_of(&self, factor: i64) -> bool {
        unsafe { mlir_affine_expr_is_multiple_of(*self, factor) }
    }

    /// Checks whether the given affine expression involves AffineDimExpr
    /// 'position'.
    fn is_function_of_dim(&self, position: usize) -> bool {
        unsafe { mlir_affine_expr_is_function_of_dim(*self, position) }
    }

    /// Composes the given map with the given expression.
    fn compose(&self, map: AffineMap) -> AffineExprBase {
        unsafe { mlir_affine_expr_compose(*self, map) }
    }
}
impl AffineExprBase {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Returns true if this type is an instance of the concrete type `T`
    #[inline(always)]
    pub fn isa<T>(self) -> bool
    where
        T: TryFrom<Self>,
    {
        T::try_from(self).is_ok()
    }

    /// Tries to convert this type to an instance of the concrete type `T`
    #[inline(always)]
    pub fn dyn_cast<T>(self) -> Result<T, InvalidTypeCastError>
    where
        T: TryFrom<Self, Error = InvalidTypeCastError>,
    {
        T::try_from(self)
    }
}
impl Display for AffineExprBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_affine_expr_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}
impl fmt::Pointer for AffineExprBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for AffineExprBase {}
impl PartialEq for AffineExprBase {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_affine_expr_equal(*self, *other) }
    }
}

extern "C" {
    #[link_name = "mlirAffineExprGetContext"]
    fn mlir_affine_expr_get_context(expr: AffineExprBase) -> Context;
    #[link_name = "mlirAffineExprEqual"]
    fn mlir_affine_expr_equal(a: AffineExprBase, b: AffineExprBase) -> bool;
    #[link_name = "mlirAffineMapExpr"]
    fn mlir_affine_expr_print(
        expr: AffineExprBase,
        callback: MlirStringCallback,
        userdata: *const c_void,
    );
    #[link_name = "mlirAffineExprDump"]
    fn mlir_affine_expr_dump(expr: AffineExprBase);
    #[link_name = "mlirAffineExprIsSymbolicOrConstant"]
    fn mlir_affine_expr_is_symbolic_or_constant(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineExprIsPureAffine"]
    fn mlir_affine_expr_is_pure_affine(expr: AffineExprBase) -> bool;
    #[link_name = "mlirAffineExprGetLargestKnownDivisor"]
    fn mlir_affine_expr_get_largest_known_divisor(expr: AffineExprBase) -> i64;
    #[link_name = "mlirAffineExprIsMultipleOf"]
    fn mlir_affine_expr_is_multiple_of(expr: AffineExprBase, factor: i64) -> bool;
    #[link_name = "mlirAffineExprIsFunctionOfDim"]
    fn mlir_affine_expr_is_function_of_dim(expr: AffineExprBase, position: usize) -> bool;
    #[link_name = "mlirAffineExprCompose"]
    fn mlir_affine_expr_compose(expr: AffineExprBase, map: AffineMap) -> AffineExprBase;
}

mod affine_expr;
mod affine_map;
pub(crate) mod attributes;
mod block;
mod context;
mod function;
mod location;
mod module;
mod operations;
mod region;
mod symbols;
pub(crate) mod types;
mod values;

pub use self::affine_expr::*;
pub use self::affine_map::*;
pub use self::attributes::*;
pub use self::block::*;
pub use self::context::*;
pub use self::function::*;
pub use self::location::*;
pub use self::module::*;
pub use self::operations::*;
pub use self::region::*;
pub use self::symbols::{SymbolTable, Visibility};
pub use self::types::*;
pub use self::values::*;

use crate::support::StringRef;
use std::ops::Deref;

/// This error is used with `TryFrom<>` for various MLIR types
#[derive(thiserror::Error, Debug)]
#[error("invalid target type for dynamic cast")]
pub struct InvalidTypeCastError;

extern "C" {
    type MlirTypeId;
    type MlirBuilder;
    type MlirOpBuilder;
    type MlirInsertPoint;
}

/// Represents the C++ equivalent of Rust type ids.
///
/// Used for type checks/casts.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct TypeId(*mut MlirTypeId);
impl TypeId {
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }
}
impl Eq for TypeId {}
impl PartialEq for TypeId {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_type_id_equal(*self, *other) }
    }
}
impl std::hash::Hash for TypeId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let value = unsafe { mlir_type_id_hash_value(*self) };
        value.hash(state);
    }
}

extern "C" {
    #[link_name = "mlirTypeIDEqual"]
    fn mlir_type_id_equal(a: TypeId, b: TypeId) -> bool;
    #[link_name = "mlirTypeIDHashValue"]
    fn mlir_type_id_hash_value(id: TypeId) -> usize;
}

/// This trait encompasses functionality common to all
/// derived `Builder` types in MLIR
///
/// Common functionality includes building types and attributes
pub trait Builder {
    /// Get the context the builder is associated with
    fn context(&self) -> Context {
        unsafe { mlir_builder_get_context(self.base()) }
    }
    /// Obtain an identifier from the given string
    ///
    /// NOTE: Identifiers are just StringAttrs in recent versions of MLIR
    fn get_identifier<S: Into<StringRef>>(&self, name: S) -> StringAttr {
        StringAttr::get(self.context(), name)
    }
    /// Obtain a `Location` value of unknown origin
    fn get_unknown_loc(&self) -> Location {
        unsafe { mlir_builder_get_unknown_loc(self.base()) }
    }
    /// Obtain a `Location` value from file/line/column info
    fn get_file_line_col_loc<S: Into<StringRef>>(
        &self,
        filename: S,
        line: u32,
        col: u32,
    ) -> Location {
        unsafe { mlir_builder_get_file_line_col_loc(self.base(), filename.into(), line, col) }
    }
    /// Fuse multiple `Location` values into a single `Location`
    fn get_fused_loc(&self, locs: &[Location]) -> Location {
        unsafe { mlir_builder_get_fused_loc(self.base(), locs.len(), locs.as_ptr()) }
    }
    /// Get a type corresponding to `f64`
    fn get_f64_type(&self) -> FloatType {
        unsafe { mlir_builder_get_f64_type(self.base()) }
    }
    /// Get a type corresponding to the opaque index type (a kind of integer)
    fn get_index_type(&self) -> IndexType {
        unsafe { mlir_builder_get_index_type(self.base()) }
    }
    /// Get a type corresponding to `i1`
    fn get_i1_type(&self) -> IntegerType {
        unsafe { mlir_builder_get_i1_type(self.base()) }
    }
    /// Get a type corresponding to `i8`
    fn get_i8_type(&self) -> IntegerType {
        self.get_integer_type(8)
    }
    /// Get a type corresponding to `i16`
    fn get_i16_type(&self) -> IntegerType {
        self.get_integer_type(16)
    }
    /// Get a type corresponding to `i32`
    fn get_i32_type(&self) -> IntegerType {
        unsafe { mlir_builder_get_i32_type(self.base()) }
    }
    /// Get a type corresponding to `i64`
    fn get_i64_type(&self) -> IntegerType {
        unsafe { mlir_builder_get_i64_type(self.base()) }
    }
    /// Get a type corresponding to a signless integer of the provided bit width
    fn get_integer_type(&self, width: u32) -> IntegerType {
        unsafe { mlir_builder_get_integer_type(self.base(), width) }
    }
    /// Get a type corresponding to a signed integer of the provided bit width
    fn get_signed_integer_type(&self, width: u32) -> IntegerType {
        unsafe { mlir_builder_get_signed_integer_type(self.base(), width) }
    }
    /// Get a type corresponding to a unsigned integer of the provided bit width
    fn get_unsigned_integer_type(&self, width: u32) -> IntegerType {
        unsafe { mlir_builder_get_unsigned_integer_type(self.base(), width) }
    }
    /// Get a type corresponding to a tuple of the given elements
    fn get_tuple_type(&self, elements: &[TypeBase]) -> TupleType {
        TupleType::get(self.context(), elements)
    }
    /// Get a type corresponding to a memref with the given element type,
    /// rank, shape, and address space, with a potentially empty list of affine maps.
    fn get_memref_type<T: Type>(
        &self,
        element_type: T,
        shape: &[u64],
        layout: AffineMapAttr,
        addrspace: AddressSpace,
    ) -> MemRefType {
        MemRefType::get(element_type, shape, layout, addrspace)
    }

    /// Get a type corresponding to a memref with the given element type, shape and address space.
    ///
    /// This is slightly different than the one returned by `get_memref_type`, as it has no affine maps, i.e.
    /// it represents a default row-major contiguous memref.
    fn get_contiguous_memref_type<T: Type>(
        element_type: T,
        shape: &[u64],
        addrspace: AddressSpace,
    ) -> MemRefType {
        MemRefType::get_contiguous(element_type, shape, addrspace)
    }

    /// Get a type corresponding to a memref of dynamic rank, with the given element type and address space.
    fn get_unranked_memref<T: Type>(element_type: T, addrspace: AddressSpace) -> MemRefType {
        MemRefType::get_unranked(element_type, addrspace)
    }
    /// Get a type corresponding to a function signature with the given input and result types
    fn get_function_type(&self, inputs: &[TypeBase], results: &[TypeBase]) -> FunctionType {
        let inputc = inputs.len();
        let inputv = inputs.as_ptr();
        let resultc = results.len();
        let resultv = results.as_ptr();
        unsafe { mlir_builder_get_function_type(self.base(), inputc, inputv, resultc, resultv) }
    }
    /// Get a type respresenting the none value, something akin to `null`
    fn get_none_type(&self) -> NoneType {
        unsafe { mlir_builder_get_none_type(self.base()) }
    }
    /// Get a type representing a 1D vector of `element_ty` and `arity`
    fn get_array_type<T: Type>(&self, element_ty: T, arity: usize) -> VectorType {
        VectorType::get(element_ty, &[arity as u64])
    }
    /// Get a type representing an N-dimensional vector of `element_ty` and `arity`
    fn get_vector_type<T: Type>(&self, element_ty: T, shape: &[u64]) -> VectorType {
        VectorType::get(element_ty, shape)
    }
    /// Associate the given name to the provided attribute value as a `NamedAttribute`
    fn get_named_attr<S: Into<StringRef>, A: Attribute>(
        &self,
        name: S,
        value: A,
    ) -> NamedAttribute {
        NamedAttribute::get(self.get_string_attr(name), value)
    }
    /// Get an attribute which has no value (i.e. its presence is significant)
    fn get_unit_attr(&self) -> UnitAttr {
        unsafe { mlir_builder_get_unit_attr(self.base()) }
    }
    /// Get an attribute which has a type value
    fn get_type_attr<T: Type>(&self, ty: T) -> TypeAttr {
        TypeAttr::get(ty)
    }
    /// Get an attribute which has a boolean value
    fn get_bool_attr(&self, value: bool) -> BoolAttr {
        unsafe { mlir_builder_get_bool_attr(self.base(), value) }
    }
    /// Get an attribute which is a collection of key/value pairs, or more precisely, a set of
    /// `NamedAttribute`
    fn get_dictionary_attr(&self, values: &[NamedAttribute]) -> DictionaryAttr {
        DictionaryAttr::get(self.context(), values)
    }
    /// Get an attribute which has an integer value of the given type
    fn get_integer_attr<I: IntegerLike>(&self, ty: I, value: i64) -> IntegerAttr {
        unsafe { mlir_builder_get_integer_attr(self.base(), ty.base(), value) }
    }
    /// Get an attribute which has an i8 integer value
    fn get_i8_attr(&self, value: i8) -> IntegerAttr {
        unsafe { mlir_builder_get_i8_attr(self.base(), value) }
    }
    /// Get an attribute which has an i16 integer value
    fn get_i16_attr(&self, value: i16) -> IntegerAttr {
        unsafe { mlir_builder_get_i16_attr(self.base(), value) }
    }
    /// Get an attribute which has an i32 integer value
    fn get_i32_attr(&self, value: i32) -> IntegerAttr {
        unsafe { mlir_builder_get_i32_attr(self.base(), value) }
    }
    /// Get an attribute which has an i64 integer value
    fn get_i64_attr(&self, value: i64) -> IntegerAttr {
        unsafe { mlir_builder_get_i64_attr(self.base(), value) }
    }
    /// Get an attribute which has an index value
    fn get_index_attr(&self, value: i64) -> IntegerAttr {
        unsafe { mlir_builder_get_index_attr(self.base(), value) }
    }
    /// Get an attribute which has an float value of the given type
    fn get_float_attr(&self, ty: FloatType, value: f64) -> FloatAttr {
        unsafe { mlir_builder_get_float_attr(self.base(), ty, value) }
    }
    /// Get an attribute which has an f64 integer value
    fn get_f64_attr(&self, value: f64) -> FloatAttr {
        self.get_float_attr(self.get_f64_type(), value)
    }
    /// Get an attribute which has a string value
    fn get_string_attr<S: Into<StringRef>>(&self, value: S) -> StringAttr {
        unsafe { mlir_builder_get_string_attr(self.base(), value.into()) }
    }
    /// Get an attribute which has an array value
    fn get_array_attr(&self, values: &[AttributeBase]) -> ArrayAttr {
        unsafe { mlir_builder_get_array_attr(self.base(), values.len(), values.as_ptr()) }
    }
    /// Get an attribute which has a FlatSymbolRef value, using the given Operation to get the
    /// symbol name
    fn get_flat_symbol_ref_attr_by_op<O: Operation>(&self, op: O) -> FlatSymbolRefAttr {
        unsafe { mlir_builder_get_flat_symbol_ref_attr_by_operation(self.base(), op.base()) }
    }
    /// Get an attribute which has a FlatSymbolRef value, using the given string as the symbol name
    fn get_flat_symbol_ref_attr_by_name<S: Into<StringRef>>(&self, symbol: S) -> FlatSymbolRefAttr {
        unsafe { mlir_builder_get_flat_symbol_ref_attr_by_name(self.base(), symbol.into()) }
    }
    /// Get an attribute which has a SymbolRef value, using the given string as the root symbol
    /// name, and the given nested SymbolRef/FlatSymbolRef values
    fn get_symbol_ref_attr<S: Into<StringRef>>(
        &self,
        value: S,
        nested: &[SymbolRefAttr],
    ) -> SymbolRefAttr {
        unsafe {
            mlir_builder_get_symbol_ref_attr(
                self.base(),
                value.into(),
                nested.len(),
                nested.as_ptr(),
            )
        }
    }
    /// Get an attribute which has the given AffineMap value
    fn get_affine_map_attr(&self, map: AffineMap) -> AffineMapAttr {
        AffineMapAttr::get(map)
    }
    /// Get a zero-result affine map with no dimensions or symbols
    fn get_empty_affine_map(&self) -> AffineMap {
        AffineMap::get_empty(self.context())
    }
    /// Creates a zero-result affine map with the given dimensions and symbols
    fn get_zero_result_affine_map(&self, dims: usize, symbols: usize) -> AffineMap {
        AffineMap::get_zero_result(self.context(), dims, symbols)
    }
    /// Get an affine map with results defined by the given list of affine expressions.
    /// The resulting map also has the requested number of input dimensions and symbols,
    /// regardless of them being used in the results.
    fn get_affine_map(&self, dims: usize, symbols: usize, exprs: &[AffineExprBase]) -> AffineMap {
        AffineMap::get(self.context(), dims, symbols, exprs)
    }
    /// Gets a single constant result affine map
    fn get_constant_affine_map(&self, value: u64) -> AffineMap {
        AffineMap::get_constant(self.context(), value)
    }
    /// Gets an affine map with `dims` identity
    fn get_multi_dim_identity_affine_map(&self, dims: usize) -> AffineMap {
        AffineMap::get_multi_dim_identity(self.context(), dims)
    }
    /// Gets an identity affine map on the most minor dimensions.
    /// This function will panic if the number of dimensions is greater or equal to the number of results
    fn get_minor_identity_affine_map(&self, dims: usize, results: usize) -> AffineMap {
        AffineMap::get_minor_identity(self.context(), dims, results)
    }
    /// Gets an affine map with a permutation expression and its size.
    ///
    /// The permutation expression is a non-empty vector of integers.
    /// The elements of the permutation vector must be continuous from 0 and cannot be repeated,
    /// i.e. `[1, 2, 0]` is a valid permutation, `[2, 0]` or `[1, 1, 2]` are not.
    fn get_affine_map_permutation(&self, permutation: Permutation) -> AffineMap {
        AffineMap::get_permutation(self.context(), permutation)
    }

    fn base(&self) -> BuilderBase;
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct BuilderBase(*mut MlirBuilder);
impl Builder for BuilderBase {
    #[inline(always)]
    fn base(&self) -> BuilderBase {
        *self
    }
}

/// This trait extends the `Builder` trait with op-building functionality
pub trait OpBuilder: Builder {
    /// Saves the current insertion point to a guard, that when dropped, restores the builder
    /// to the saved location
    fn insertion_guard<'a>(&'a self) -> InsertionGuard<'a> {
        let builder = self.base().into();
        let ip = unsafe { mlir_op_builder_save_insertion_point(builder) };
        InsertionGuard {
            ip,
            builder,
            _marker: core::marker::PhantomData,
        }
    }
    /// Sets the insertion point to just before the given operation
    fn set_insertion_point<O: Operation>(&self, op: O) {
        unsafe {
            mlir_op_builder_set_insertion_point(self.base().into(), op.base());
        }
    }
    /// Sets the insertion point to just after the given operation
    fn set_insertion_point_after<O: Operation>(&self, op: O) {
        unsafe {
            mlir_op_builder_set_insertion_point_after(self.base().into(), op.base());
        }
    }
    /// Sets the insertion point to just after the operation that introduced the given value
    fn set_insertion_point_after_value<V: Value>(&self, value: V) {
        unsafe {
            mlir_op_builder_set_insertion_point_after_value(self.base().into(), value.base());
        }
    }
    /// Sets the insertion point to the start of the given block
    fn set_insertion_point_to_start(&self, block: Block) {
        unsafe {
            mlir_op_builder_set_insertion_point_to_start(self.base().into(), block);
        }
    }
    /// Sets the insertion point to the end of the given block
    fn set_insertion_point_to_end(&self, block: Block) {
        unsafe {
            mlir_op_builder_set_insertion_point_to_end(self.base().into(), block);
        }
    }
    /// Returns the current block the builder is inserting into, if positioned inside a block
    fn get_insertion_block(&self) -> Option<Block> {
        let block = unsafe { mlir_op_builder_get_insertion_block(self.base().into()) };
        if block.is_null() {
            None
        } else {
            Some(block)
        }
    }
    /// Creates a new block with the given arguments, appending it to the given region
    fn create_block_in_region(
        &self,
        region: Region,
        args: &[TypeBase],
        locs: &[Location],
    ) -> Block {
        assert_eq!(args.len(), locs.len());
        unsafe {
            mlir_op_builder_create_block(
                self.base().into(),
                region,
                args.len(),
                args.as_ptr(),
                locs.as_ptr(),
            )
        }
    }
    /// Creates a new block with the given arguments, placing it before the given block
    fn create_block_before(&self, before: Block, args: &[TypeBase], locs: &[Location]) -> Block {
        assert_eq!(args.len(), locs.len());
        unsafe {
            mlir_op_builder_create_block_before(
                self.base().into(),
                before,
                args.len(),
                args.as_ptr(),
                locs.as_ptr(),
            )
        }
    }
    /// Inserts the given operation
    fn insert_operation<O: Operation>(&self, op: O) -> OperationBase {
        unsafe { mlir_op_builder_insert_operation(self.base().into(), op.base()) }
    }
    /// Creates an operation of the given type using the provided state
    ///
    /// This is a low-level operation, and should be used by dialect-specific builders
    /// to construct their operations.
    fn create_operation<O: Operation>(&self, state: OperationState) -> anyhow::Result<O>
    where
        O: Operation + TryFrom<OperationBase, Error = InvalidTypeCastError>,
    {
        let base = unsafe { mlir_op_builder_create_operation(self.base().into(), &state) };
        if base.is_null() {
            Err(InvalidOperationError.into())
        } else {
            O::try_from(base).map_err(|e| e.into())
        }
    }
    /// Clones the given operation
    fn clone_operation<O: Operation>(&self, op: O) -> O
    where
        O: Operation + TryFrom<OperationBase, Error = InvalidTypeCastError>,
    {
        let base = unsafe { mlir_op_builder_clone_operation(self.base().into(), op.base()) };
        assert!(!base.is_null(), "clone of operation returned null");
        O::try_from(base).unwrap()
    }
    /// Creates a new ModuleOp with the given source location and name
    ///
    /// NOTE: Ownership is with the caller
    fn create_module<S: Into<StringRef>>(&self, loc: Location, name: S) -> OwnedModule {
        unsafe { mlir_op_builder_create_module(self.base().into(), loc, name.into()) }
    }
    /// Gets the function with the given symbol name, using the symbol table associated with
    /// the operation currently being inserted into.
    ///
    /// If the symbol is found, but not associated with a FuncOp, this function will panic
    ///
    /// If the current operation's symbol table does not contain an entry by that name, returns None
    fn get_func_by_symbol<S: Into<StringRef>>(&self, name: S) -> Option<FuncOp> {
        let current_block = self.get_insertion_block()?;
        let current_op = current_block.operation().unwrap();
        if let Some(op) = SymbolTable::lookup_nearest_symbol_from(current_op, name) {
            Some(op.try_into().unwrap())
        } else {
            None
        }
    }
    /// Declares a function with the given source location, name, type, visibility and attributes,
    /// unless a function by that name already exists, in which case the existing declaration is
    /// returned instead.
    ///
    /// This function will panic if the symbol name provided is associated with an operation that is
    /// not a FuncOp.
    fn get_or_declare_func<S: Into<StringRef>>(
        &self,
        loc: Location,
        name: S,
        ty: FunctionType,
        vis: Visibility,
        attrs: &[NamedAttribute],
    ) -> FuncOp {
        let name = name.into();
        if let Some(op) = self.get_func_by_symbol(name) {
            op
        } else {
            let func = self.build_func(loc, name, ty, attrs, &[]);
            func.set_visibility(vis);
            func
        }
    }
    /// Builds a FuncOp with the given source location, name, type, and attributes
    fn build_func<S: Into<StringRef>>(
        &self,
        loc: Location,
        name: S,
        ty: FunctionType,
        attrs: &[NamedAttribute],
        arg_attrs: &[DictionaryAttr],
    ) -> FuncOp {
        unsafe {
            mlir_func_build_func_op(
                self.base().into(),
                loc,
                name.into(),
                ty,
                attrs.len(),
                attrs.as_ptr(),
                arg_attrs.len(),
                arg_attrs.as_ptr(),
            )
        }
    }
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct OpBuilderBase(*mut MlirOpBuilder);
impl Builder for OpBuilderBase {
    #[inline(always)]
    fn base(&self) -> BuilderBase {
        (*self).into()
    }
}
impl OpBuilder for OpBuilderBase {}
impl Into<BuilderBase> for OpBuilderBase {
    fn into(self) -> BuilderBase {
        BuilderBase(unsafe { std::mem::transmute::<*mut MlirOpBuilder, *mut MlirBuilder>(self.0) })
    }
}
impl From<BuilderBase> for OpBuilderBase {
    fn from(builder: BuilderBase) -> Self {
        Self(unsafe { std::mem::transmute::<*mut MlirBuilder, *mut MlirOpBuilder>(builder.0) })
    }
}

/// This type is essentially the concrete implementation of the `Builder`
/// trait, but in C++ this type corresponds to `mlir::Builder`, which is
/// the base class for the more useful `OpBuilder` class. We model it here
/// in case in the future we need to support APIs that only take a `Builder`,
/// not `OpBuilder` in MLIR.
#[repr(transparent)]
pub struct OwnedBuilder(BuilderBase);
impl OwnedBuilder {
    pub fn new(context: Context) -> Self {
        unsafe { mlir_builder_new_from_context(context) }
    }

    pub fn from_module(module: Module) -> Self {
        unsafe { mlir_builder_new_from_module(module) }
    }
}
impl Builder for OwnedBuilder {
    #[inline]
    fn base(&self) -> BuilderBase {
        self.0
    }
}
impl Deref for OwnedBuilder {
    type Target = BuilderBase;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The `OpBuilder` type is the primary builder type used when constructing
/// IR during compilation. With it, one can do all the things `Builder` can do,
/// but can also construct operations, blocks, and regions.
#[repr(transparent)]
pub struct OwnedOpBuilder(OpBuilderBase);
impl OwnedOpBuilder {
    #[inline]
    pub fn new(context: Context) -> Self {
        unsafe { mlir_op_builder_new_from_context(context) }
    }

    #[inline]
    pub fn at_block_start(block: Block) -> Self {
        unsafe { mlir_op_builder_at_block_begin(block) }
    }

    #[inline]
    pub fn at_block_end(block: Block) -> Self {
        unsafe { mlir_op_builder_at_block_end(block) }
    }

    #[inline]
    pub fn at_block_terminator(block: Block) -> Self {
        unsafe { mlir_op_builder_at_block_terminator(block) }
    }
}
impl Drop for OwnedOpBuilder {
    fn drop(&mut self) {
        unsafe { mlir_op_builder_destroy(self.0) }
    }
}
impl Builder for OwnedOpBuilder {
    fn base(&self) -> BuilderBase {
        self.0.into()
    }
}
impl OpBuilder for OwnedOpBuilder {}
impl Deref for OwnedOpBuilder {
    type Target = OpBuilderBase;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

extern "C" {
    #[link_name = "mlirBuilderNewFromContext"]
    fn mlir_builder_new_from_context(context: Context) -> OwnedBuilder;
    #[link_name = "mlirBuilderNewFromModule"]
    fn mlir_builder_new_from_module(module: Module) -> OwnedBuilder;
    #[link_name = "mlirBuilderGetContext"]
    fn mlir_builder_get_context(builder: BuilderBase) -> Context;

    #[link_name = "mlirBuilderGetUnknownLoc"]
    fn mlir_builder_get_unknown_loc(builder: BuilderBase) -> Location;
    #[link_name = "mlirBuilderGetFileLineColLoc"]
    fn mlir_builder_get_file_line_col_loc(
        builder: BuilderBase,
        filename: StringRef,
        line: u32,
        column: u32,
    ) -> Location;
    #[link_name = "mlirBuilderGetFusedLoc"]
    fn mlir_builder_get_fused_loc(
        builder: BuilderBase,
        num_locs: usize,
        locs: *const Location,
    ) -> Location;
    #[link_name = "mlirBuilderGetF64Type"]
    fn mlir_builder_get_f64_type(builder: BuilderBase) -> FloatType;
    #[link_name = "mlirBuilderGetIndexType"]
    fn mlir_builder_get_index_type(builder: BuilderBase) -> IndexType;
    #[link_name = "mlirBuilderGetI1Type"]
    fn mlir_builder_get_i1_type(builder: BuilderBase) -> IntegerType;
    #[link_name = "mlirBuilderGetI32Type"]
    fn mlir_builder_get_i32_type(builder: BuilderBase) -> IntegerType;
    #[link_name = "mlirBuilderGetI64Type"]
    fn mlir_builder_get_i64_type(builder: BuilderBase) -> IntegerType;
    #[link_name = "mlirBuilderGetIntegerType"]
    fn mlir_builder_get_integer_type(builder: BuilderBase, width: u32) -> IntegerType;
    #[link_name = "mlirBuilderGetSignedIntegerType"]
    fn mlir_builder_get_signed_integer_type(builder: BuilderBase, width: u32) -> IntegerType;
    #[link_name = "mlirBuilderGetUnsignedIntegerType"]
    fn mlir_builder_get_unsigned_integer_type(builder: BuilderBase, width: u32) -> IntegerType;
    #[link_name = "mlirBuilderGetFunctionType"]
    fn mlir_builder_get_function_type(
        builder: BuilderBase,
        num_inputs: usize,
        inputs: *const TypeBase,
        num_results: usize,
        results: *const TypeBase,
    ) -> FunctionType;
    #[link_name = "mlirBuilderGetNoneType"]
    fn mlir_builder_get_none_type(builder: BuilderBase) -> NoneType;

    #[link_name = "mlirBuilderGetUnitAttr"]
    fn mlir_builder_get_unit_attr(builder: BuilderBase) -> UnitAttr;
    #[link_name = "mlirBuilderGetBoolAttr"]
    fn mlir_builder_get_bool_attr(builder: BuilderBase, value: bool) -> BoolAttr;
    #[link_name = "mlirBuilderGetIntegerAttr"]
    fn mlir_builder_get_integer_attr(builder: BuilderBase, ty: TypeBase, value: i64)
        -> IntegerAttr;
    #[link_name = "mlirBuilderGetI8Attr"]
    fn mlir_builder_get_i8_attr(builder: BuilderBase, value: i8) -> IntegerAttr;
    #[link_name = "mlirBuilderGetI16Attr"]
    fn mlir_builder_get_i16_attr(builder: BuilderBase, value: i16) -> IntegerAttr;
    #[link_name = "mlirBuilderGetI32Attr"]
    fn mlir_builder_get_i32_attr(builder: BuilderBase, value: i32) -> IntegerAttr;
    #[link_name = "mlirBuilderGetI64Attr"]
    fn mlir_builder_get_i64_attr(builder: BuilderBase, value: i64) -> IntegerAttr;
    #[link_name = "mlirBuilderGetIndexAttr"]
    fn mlir_builder_get_index_attr(builder: BuilderBase, value: i64) -> IntegerAttr;
    #[link_name = "mlirBuilderGetFloatAttr"]
    fn mlir_builder_get_float_attr(builder: BuilderBase, ty: FloatType, value: f64) -> FloatAttr;
    #[link_name = "mlirBuilderGetStringAttr"]
    fn mlir_builder_get_string_attr(builder: BuilderBase, value: StringRef) -> StringAttr;
    #[link_name = "mlirBuilderGetArrayAttr"]
    fn mlir_builder_get_array_attr(
        builder: BuilderBase,
        num_elements: usize,
        elements: *const AttributeBase,
    ) -> ArrayAttr;
    #[link_name = "mlirBuilderGetFlatSymbolRefAttrByOperation"]
    fn mlir_builder_get_flat_symbol_ref_attr_by_operation(
        builder: BuilderBase,
        op: OperationBase,
    ) -> FlatSymbolRefAttr;
    #[link_name = "mlirBuilderGetFlatSymbolRefAttrByName"]
    fn mlir_builder_get_flat_symbol_ref_attr_by_name(
        builder: BuilderBase,
        symbol: StringRef,
    ) -> FlatSymbolRefAttr;
    #[link_name = "mlirBuilderGetSymbolRefAttr"]
    fn mlir_builder_get_symbol_ref_attr(
        builder: BuilderBase,
        value: StringRef,
        num_nested: usize,
        nested: *const SymbolRefAttr,
    ) -> SymbolRefAttr;

    #[link_name = "mlirOpBuilderNewFromContext"]
    fn mlir_op_builder_new_from_context(context: Context) -> OwnedOpBuilder;
    #[link_name = "mlirOpBuilderCreateModule"]
    fn mlir_op_builder_create_module(
        builder: OpBuilderBase,
        loc: Location,
        name: StringRef,
    ) -> OwnedModule;
    #[link_name = "mlirOpBuilderAtBlockBegin"]
    fn mlir_op_builder_at_block_begin(block: Block) -> OwnedOpBuilder;
    #[link_name = "mlirOpBuilderAtBlockEnd"]
    fn mlir_op_builder_at_block_end(block: Block) -> OwnedOpBuilder;
    #[link_name = "mlirOpBuilderAtBlockTerminator"]
    fn mlir_op_builder_at_block_terminator(block: Block) -> OwnedOpBuilder;
    #[link_name = "mlirOpBuilderDestroy"]
    fn mlir_op_builder_destroy(builder: OpBuilderBase);
    #[link_name = "mlirOpBuilderSetInsertionPoint"]
    fn mlir_op_builder_set_insertion_point(builder: OpBuilderBase, op: OperationBase);
    #[link_name = "mlirOpBuilderSetInsertionPointAfter"]
    fn mlir_op_builder_set_insertion_point_after(builder: OpBuilderBase, op: OperationBase);
    #[link_name = "mlirOpBuilderSetInsertionPointAfterValue"]
    fn mlir_op_builder_set_insertion_point_after_value(builder: OpBuilderBase, value: ValueBase);
    #[link_name = "mlirOpBuilderSetInsertionPointToStart"]
    fn mlir_op_builder_set_insertion_point_to_start(builder: OpBuilderBase, block: Block);
    #[link_name = "mlirOpBuilderSetInsertionPointToEnd"]
    fn mlir_op_builder_set_insertion_point_to_end(builder: OpBuilderBase, block: Block);
    #[link_name = "mlirOpBuilderGetInsertionBlock"]
    fn mlir_op_builder_get_insertion_block(builder: OpBuilderBase) -> Block;
    #[link_name = "mlirOpBuilderCreateBlock"]
    fn mlir_op_builder_create_block(
        builder: OpBuilderBase,
        parent: Region,
        num_args: usize,
        args: *const TypeBase,
        locs: *const Location,
    ) -> Block;
    #[link_name = "mlirOpBuilderCreateBlockBefore"]
    fn mlir_op_builder_create_block_before(
        builder: OpBuilderBase,
        before: Block,
        num_args: usize,
        args: *const TypeBase,
        locs: *const Location,
    ) -> Block;
    #[link_name = "mlirOpBuilderInsertOperation"]
    fn mlir_op_builder_insert_operation(builder: OpBuilderBase, op: OperationBase)
        -> OperationBase;
    #[link_name = "mlirOpBuilderCreateOperation"]
    fn mlir_op_builder_create_operation(
        builder: OpBuilderBase,
        state: *const OperationState,
    ) -> OperationBase;
    #[link_name = "mlirOpBuilderCloneOperation"]
    fn mlir_op_builder_clone_operation(builder: OpBuilderBase, op: OperationBase) -> OperationBase;
}

/// This opaque type represents the insertion point of an OpBuilder
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct InsertPoint(*mut MlirInsertPoint);

/// This type provides an RAII-style guard which saves the insertion
/// point of its parent OpBuilder on creation, and restores the saved
/// insertion point when dropped.
pub struct InsertionGuard<'a> {
    ip: InsertPoint,
    builder: OpBuilderBase,
    _marker: core::marker::PhantomData<&'a OpBuilderBase>,
}
impl Drop for InsertionGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            mlir_op_builder_restore_insertion_point(self.builder, self.ip);
        }
    }
}

extern "C" {
    #[link_name = "mlirOpBuilderSaveInsertionPoint"]
    fn mlir_op_builder_save_insertion_point(builder: OpBuilderBase) -> InsertPoint;
    #[link_name = "mlirOpBuilderRestoreInsertionPoint"]
    fn mlir_op_builder_restore_insertion_point(builder: OpBuilderBase, ip: InsertPoint);
}

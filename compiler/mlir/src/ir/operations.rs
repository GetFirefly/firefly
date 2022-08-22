use std::ffi::c_void;
use std::fmt::{self, Display};
use std::iter::FusedIterator;

use crate::support::{self, LogicalResult, MlirStringCallback, StringRef};
use crate::{Context, SymbolTable, Visibility};

use super::*;

extern "C" {
    pub(crate) type MlirOperation;
}

/// Indicates that an API was given/returned an invalid (null) OperationBase handle
#[derive(thiserror::Error, Debug)]
#[error("invalid operation, expected non-null reference")]
pub struct InvalidOperationError;

/// This trait is implemented by all concrete implementations of an MLIR operation in Rust.
pub trait Operation {
    /// Returns the context in which this Operation was created
    fn context(&self) -> Context {
        unsafe { mlir_operation_get_context(self.base()) }
    }
    /// Returns the name of this operation as an identifier/StringAttr
    fn name(&self) -> StringAttr {
        unsafe { mlir_operation_get_name(self.base()) }
    }
    /// Returns the dialect namespace in which this operation is defined
    fn dialect_name(&self) -> StringRef {
        unsafe { mlir_operation_get_dialect_name(self.base()) }
    }
    /// Verifies this operation and any nested operations to ensure invariants are upheld
    ///
    /// NOTE: This can be expensive, so it should only be used for dev/debugging
    fn is_valid(&self) -> bool {
        match unsafe { mlir_operation_verify(self.base()) } {
            LogicalResult::Success => true,
            _ => false,
        }
    }
    /// Gets the block containing this operation, if there is one
    fn get_block(&self) -> Option<Block> {
        let block = unsafe { mlir_operation_get_block(self.base()) };
        if block.is_null() {
            None
        } else {
            Some(block)
        }
    }
    /// Gets the parent operation which contains this operation, if there is one
    fn get_parent(&self) -> Option<OperationBase> {
        let op = unsafe { mlir_operation_get_parent_operation(self.base()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }
    /// Returns the module in which this operation is defined, if it was defined in one
    fn get_parent_module(&self) -> Option<Module> {
        let op = unsafe { mlir_operation_get_parent_module(self.base()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }
    /// Gets the number of regions in this operation
    fn num_regions(&self) -> usize {
        unsafe { mlir_operation_get_num_regions(self.base()) }
    }
    /// Gets the region at the given index
    ///
    /// NOTE: This function will panic if the index is out of bounds
    fn get_region(&self, index: usize) -> Region {
        let region = unsafe { mlir_operation_get_region(self.base(), index) };
        assert!(
            !region.is_null(),
            "invalid region index {}, out of bounds",
            index
        );
        region
    }
    /// Gets the entry block in the first region of this operation, if it contains one
    ///
    /// NOTE: This function will panic if this op has no regions
    fn get_body(&self) -> Option<Block> {
        self.get_region(0).entry()
    }
    /// Gets the next operation following this one in its containing region, if there is one
    fn next(&self) -> Option<OperationBase> {
        let op = unsafe { mlir_operation_get_next_in_block(self.base()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }
    /// Gets the number of operands provided to this operation
    fn num_operands(&self) -> usize {
        unsafe { mlir_operation_get_num_operands(self.base()) }
    }
    /// Gets the operand at the given index
    ///
    /// NOTE: This function will panic if the index is out of bounds
    fn get_operand(&self, index: usize) -> ValueBase {
        let val = unsafe { mlir_operation_get_operand(self.base(), index) };
        assert!(
            !val.is_null(),
            "invalid operand index {}, out of bounds",
            index
        );
        val
    }
    /// Returns an iterator over the operation operands
    fn operands(&self) -> OpOperandIter {
        OpOperandIter::new(self.base())
    }
    /// Gets the number of results produced by this operation
    fn num_results(&self) -> usize {
        unsafe { mlir_operation_get_num_results(self.base()) }
    }
    /// Gets the result at the given index
    ///
    /// NOTE: This function will panic if the index is out of bounds
    fn get_result(&self, index: usize) -> OpResult {
        let val = unsafe { mlir_operation_get_result(self.base(), index) };
        assert!(
            !val.is_null(),
            "invalid result index {}, out of bounds",
            index
        );
        val
    }
    /// Returns an iterator over the operation operands
    fn results(&self) -> OpResultIter {
        OpResultIter::new(self.base())
    }
    /// Returns the number of successor blocks reachable from this operation
    fn num_successors(&self) -> usize {
        unsafe { mlir_operation_get_num_successors(self.base()) }
    }
    /// Gets the successor block at the given index
    ///
    /// NOTE: This function will panic if the index is out of bounds
    fn get_successor(&self, index: usize) -> Block {
        let block = unsafe { mlir_operation_get_successor(self.base(), index) };
        assert!(
            !block.is_null(),
            "invalid block index {}, out of bounds",
            index
        );
        block
    }
    /// Gets an iterator of the successor blocks of this operation
    fn successors(&self) -> OpSuccessorIter {
        OpSuccessorIter::new(self.base())
    }
    /// Returns the number of attributes this operation has
    fn num_attributes(&self) -> usize {
        unsafe { mlir_operation_get_num_attributes(self.base()) }
    }
    /// Gets the attribute at the given index
    ///
    /// NOTE: This function will panic if the index is out of bounds
    fn get_attribute(&self, index: usize) -> NamedAttribute {
        let attr = unsafe { mlir_operation_get_attribute(self.base(), index) };
        assert!(
            !attr.is_null(),
            "invalid attribute index {}, out of bounds",
            index
        );
        attr
    }
    /// Gets an iterator of the named attributes associated with this operation
    fn attributes(&self) -> OpAttrIter {
        OpAttrIter::new(self.base())
    }
    /// Gets the attribute with the given name, if it exists
    fn get_attribute_by_name<S: Into<StringRef>>(&self, name: S) -> Option<AttributeBase> {
        let attr = unsafe { mlir_operation_get_attribute_by_name(self.base(), name.into()) };
        if attr.is_null() {
            None
        } else {
            Some(attr)
        }
    }
    /// Sets the attribute with the given name, if it exists
    fn set_attribute_by_name<S: Into<StringRef>, A: Attribute>(&self, name: S, attr: A) {
        unsafe {
            mlir_operation_set_attribute_by_name(self.base(), name.into(), attr.base());
        }
    }
    /// Removes the attribute with the given name, if it exists
    fn remove_attribute_by_name<S: Into<StringRef>>(&self, name: S) {
        unsafe {
            mlir_operation_remove_attribute_by_name(self.base(), name.into());
        }
    }
    /// Returns true if this Operation has a symbol
    fn is_symbol(&self) -> bool {
        self.get_attribute_by_name(SymbolTable::get_symbol_visibility_name())
            .is_some()
    }
    /// Returns the visibility of this symbol (if this operation is a symbol)
    fn visibility(&self) -> Option<Visibility> {
        if self.is_symbol() {
            Some(SymbolTable::get_symbol_visibility(self.base()))
        } else {
            None
        }
    }
    /// Dump the textual representation of this operation to stderr
    fn dump(&self) {
        unsafe {
            mlir_operation_dump(self.base());
        }
    }
    /// Walk the operations in the current block starting from the current one
    fn iter(&self) -> OperationIter {
        let op = self.base();
        OperationIter::new(if op.is_null() { None } else { Some(op) })
    }
    /// Return the OperationBase value underlying this operation
    fn base(&self) -> OperationBase;
}

/// Represents any MLIR operation type.
///
/// Corresponds to MLIR's `Operation` class
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct OperationBase(*mut MlirOperation);
impl From<*mut MlirOperation> for OperationBase {
    #[inline(always)]
    fn from(ptr: *mut MlirOperation) -> Self {
        Self(ptr)
    }
}
impl Operation for OperationBase {
    fn base(&self) -> OperationBase {
        *self
    }
}
impl OperationBase {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    #[inline(always)]
    pub fn isa<T>(self) -> bool
    where
        T: TryFrom<OperationBase>,
    {
        T::try_from(self).is_ok()
    }

    #[inline(always)]
    pub fn dyn_cast<T>(self) -> Result<T, InvalidTypeCastError>
    where
        T: TryFrom<OperationBase, Error = InvalidTypeCastError>,
    {
        T::try_from(self)
    }
}
impl fmt::Debug for OperationBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_operation_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}
impl fmt::Pointer for OperationBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for OperationBase {}
impl PartialEq for OperationBase {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_operation_equal(*self, *other) }
    }
}
impl Display for OperationBase {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_operation_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}

/// Represents an owned reference to an Operation
#[repr(transparent)]
pub struct OwnedOperation(OperationBase);
impl OwnedOperation {
    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn release(self) -> OperationBase {
        let op = self.0;
        std::mem::forget(self);
        op
    }
}
impl Operation for OwnedOperation {
    fn base(&self) -> OperationBase {
        self.0
    }
}
impl Drop for OwnedOperation {
    fn drop(&mut self) {
        unsafe {
            mlir_operation_destroy(self.0);
        }
    }
}

extern "C" {
    #[link_name = "mlirOperationDestroy"]
    fn mlir_operation_destroy(op: OperationBase);
    #[link_name = "mlirOperationEqual"]
    fn mlir_operation_equal(a: OperationBase, b: OperationBase) -> bool;
    #[link_name = "mlirOperationGetName"]
    fn mlir_operation_get_name(op: OperationBase) -> StringAttr;
    #[link_name = "mlirOperationGetBlock"]
    fn mlir_operation_get_block(op: OperationBase) -> Block;
    #[link_name = "mlirOperationGetParentOperation"]
    fn mlir_operation_get_parent_operation(op: OperationBase) -> OperationBase;
    #[link_name = "mlirOperationGetParentModule"]
    fn mlir_operation_get_parent_module(op: OperationBase) -> Module;
    #[link_name = "mlirOperationGetNumRegions"]
    fn mlir_operation_get_num_regions(op: OperationBase) -> usize;
    #[link_name = "mlirOperationGetRegion"]
    fn mlir_operation_get_region(op: OperationBase, index: usize) -> Region;
    #[link_name = "mlirOperationGetNextInBlock"]
    fn mlir_operation_get_next_in_block(op: OperationBase) -> OperationBase;
    #[link_name = "mlirOperationGetNumOperands"]
    fn mlir_operation_get_num_operands(op: OperationBase) -> usize;
    #[link_name = "mlirOperationGetOperand"]
    fn mlir_operation_get_operand(op: OperationBase, index: usize) -> ValueBase;
    #[link_name = "mlirOperationGetNumResults"]
    fn mlir_operation_get_num_results(op: OperationBase) -> usize;
    #[link_name = "mlirOperationGetResult"]
    fn mlir_operation_get_result(op: OperationBase, index: usize) -> OpResult;
    #[link_name = "mlirOperationGetNumSuccessors"]
    fn mlir_operation_get_num_successors(op: OperationBase) -> usize;
    #[link_name = "mlirOperationGetSuccessor"]
    fn mlir_operation_get_successor(op: OperationBase, index: usize) -> Block;
    #[link_name = "mlirOperationGetNumAttributes"]
    fn mlir_operation_get_num_attributes(op: OperationBase) -> usize;
    #[link_name = "mlirOperationGetAttribute"]
    fn mlir_operation_get_attribute(op: OperationBase, index: usize) -> NamedAttribute;
    #[link_name = "mlirOperationGetAttributeByName"]
    fn mlir_operation_get_attribute_by_name(op: OperationBase, name: StringRef) -> AttributeBase;
    #[link_name = "mlirOperationSetAttributeByName"]
    fn mlir_operation_set_attribute_by_name(
        op: OperationBase,
        name: StringRef,
        attr: AttributeBase,
    );
    #[link_name = "mlirOperationRemoveAttributeByName"]
    fn mlir_operation_remove_attribute_by_name(op: OperationBase, name: StringRef) -> bool;
    #[link_name = "mlirOperationPrint"]
    fn mlir_operation_print(
        op: OperationBase,
        callback: MlirStringCallback,
        userdata: *const c_void,
    );
    #[link_name = "mlirOperationDump"]
    fn mlir_operation_dump(op: OperationBase);
}

/// OperationState represents the intermediate state of an operation
/// under construction. It is the means by which we are able to construct
/// arbitrary MLIR operations without concrete types for each one.
///
/// For convenience, we provide those "wrapper" types in dialect-specific
/// modules within this crate.
#[repr(C)]
pub struct OperationState {
    name: StringRef,
    loc: Location,
    num_results: usize,
    results: *const TypeBase,
    num_operands: usize,
    operands: *const ValueBase,
    num_regions: usize,
    regions: *const Region,
    num_successors: usize,
    successors: *const Block,
    num_attributes: usize,
    attributes: *const NamedAttribute,
    enable_result_type_inference: bool,
}
impl OperationState {
    /// Creates a new operation state with the given name and location
    pub fn get<S: Into<StringRef>>(name: S, loc: Location) -> Self {
        unsafe { mlir_operation_state_get(name.into(), loc) }
    }

    /// Creates an Operation using the current OperationState
    pub fn create(self) -> OwnedOperation {
        let op = unsafe { mlir_operation_create(&self) };
        assert!(!op.is_null(), "operation validation error");
        op
    }

    /// Returns the name of this operation
    #[inline]
    pub fn name(&self) -> &StringRef {
        &self.name
    }

    /// Returns the location of this operation
    #[inline]
    pub fn location(&self) -> Location {
        self.loc
    }

    /// Returns a slice containing the result types associated with this operation
    #[inline]
    pub fn results(&self) -> &[TypeBase] {
        unsafe { core::slice::from_raw_parts(self.results, self.num_results) }
    }

    /// Adds the given result types to this operation
    pub fn add_results(&mut self, results: &[TypeBase]) {
        unsafe {
            mlir_operation_state_add_results(self, results.len(), results.as_ptr());
        }
    }

    /// Returns a slice containing the operands passed to this operation
    #[inline]
    pub fn operands(&self) -> &[ValueBase] {
        unsafe { core::slice::from_raw_parts(self.operands, self.num_operands) }
    }

    /// Adds the given operands to this operation
    pub fn add_operands(&mut self, operands: &[ValueBase]) {
        unsafe {
            mlir_operation_state_add_operands(self, operands.len(), operands.as_ptr());
        }
    }

    /// Returns a slice containing the regions this operation contains
    #[inline]
    pub fn regions(&self) -> &[Region] {
        unsafe { core::slice::from_raw_parts(self.regions, self.num_regions) }
    }

    /// Adds the given regions to this operation
    pub fn add_owned_regions(&mut self, regions: &[Region]) {
        unsafe {
            mlir_operation_state_add_owned_regions(self, regions.len(), regions.as_ptr());
        }
    }

    /// Returns a slice containing the successor blocks of this operation
    #[inline]
    pub fn successors(&self) -> &[Block] {
        unsafe { core::slice::from_raw_parts(self.successors, self.num_successors) }
    }

    /// Adds the given successors to this operation
    pub fn add_successors(&mut self, successors: &[Block]) {
        unsafe {
            mlir_operation_state_add_successors(self, successors.len(), successors.as_ptr());
        }
    }

    /// Returns a slice containing the attributes associated to this operation
    #[inline]
    pub fn attributes(&self) -> &[NamedAttribute] {
        unsafe { core::slice::from_raw_parts(self.attributes, self.num_attributes) }
    }

    /// Adds the given attributes to this operation
    pub fn add_attributes(&mut self, attrs: &[NamedAttribute]) {
        unsafe {
            mlir_operation_state_add_attributes(self, attrs.len(), attrs.as_ptr());
        }
    }

    #[inline]
    pub fn enable_result_type_inference(&mut self) {
        self.enable_result_type_inference = true;
    }

    #[inline]
    pub fn disable_result_type_inference(&mut self) {
        self.enable_result_type_inference = false;
    }
}

extern "C" {
    #[link_name = "mlirOperationGetContext"]
    fn mlir_operation_get_context(op: OperationBase) -> Context;
    #[link_name = "mlirOperationStateGet"]
    fn mlir_operation_state_get(name: StringRef, loc: Location) -> OperationState;
    #[link_name = "mlirOperationGetDialectName"]
    fn mlir_operation_get_dialect_name(op: OperationBase) -> StringRef;
    #[link_name = "mlirOperationStateAddResults"]
    fn mlir_operation_state_add_results(
        state: &mut OperationState,
        len: usize,
        results: *const TypeBase,
    );
    #[link_name = "mlirOperationStateAddOperands"]
    fn mlir_operation_state_add_operands(
        state: &mut OperationState,
        len: usize,
        operands: *const ValueBase,
    );
    #[link_name = "mlirOperationStateAddOwnedRegions"]
    fn mlir_operation_state_add_owned_regions(
        state: &mut OperationState,
        len: usize,
        operands: *const Region,
    );
    #[link_name = "mlirOperationStateAddSuccessors"]
    fn mlir_operation_state_add_successors(
        state: &mut OperationState,
        len: usize,
        succesors: *const Block,
    );
    #[link_name = "mlirOperationStateAddAttributes"]
    fn mlir_operation_state_add_attributes(
        state: &mut OperationState,
        len: usize,
        attrs: *const NamedAttribute,
    );
    #[link_name = "mlirOperationCreate"]
    fn mlir_operation_create(state: *const OperationState) -> OwnedOperation;
    #[link_name = "mlirOperationVerify"]
    fn mlir_operation_verify(op: OperationBase) -> LogicalResult;
}

// For internal use only, provides common methods for iterators on operands/results
struct OpElementIter {
    op: OperationBase,
    len: usize,
    pos: usize,
}
impl OpElementIter {
    #[inline(always)]
    fn get_operand(&self, index: usize) -> ValueBase {
        self.op.get_operand(index)
    }

    #[inline(always)]
    fn get_result(&self, index: usize) -> OpResult {
        self.op.get_result(index)
    }

    #[inline(always)]
    fn get_attribute(&self, index: usize) -> NamedAttribute {
        self.op.get_attribute(index)
    }

    #[inline(always)]
    fn get_successor(&self, index: usize) -> Block {
        self.op.get_successor(index)
    }

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.len - self.pos > 0 {
            let pos = self.pos;
            self.pos += 1;
            Some(pos)
        } else {
            None
        }
    }
}

/// Iterator over operands of an operation
pub struct OpOperandIter(OpElementIter);
impl OpOperandIter {
    fn new(op: OperationBase) -> Self {
        let len = if op.is_null() { 0 } else { op.num_operands() };
        Self(OpElementIter { op, len, pos: 0 })
    }
}
impl Iterator for OpOperandIter {
    type Item = ValueBase;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|pos| self.0.get_operand(pos))
    }
}
impl FusedIterator for OpOperandIter {}

/// Iterator over results of an operation
pub struct OpResultIter(OpElementIter);
impl OpResultIter {
    fn new(op: OperationBase) -> Self {
        let len = if op.is_null() { 0 } else { op.num_results() };
        Self(OpElementIter { op, len, pos: 0 })
    }
}
impl Iterator for OpResultIter {
    type Item = OpResult;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|pos| self.0.get_result(pos))
    }
}
impl FusedIterator for OpResultIter {}

/// Iterator over successor blocks of an operation
pub struct OpSuccessorIter(OpElementIter);
impl OpSuccessorIter {
    fn new(op: OperationBase) -> Self {
        let len = if op.is_null() { 0 } else { op.num_successors() };
        Self(OpElementIter { op, len, pos: 0 })
    }
}
impl Iterator for OpSuccessorIter {
    type Item = Block;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|pos| self.0.get_successor(pos))
    }
}
impl FusedIterator for OpSuccessorIter {}

/// Iterator over named attributes associated to an operation
pub struct OpAttrIter(OpElementIter);
impl OpAttrIter {
    fn new(op: OperationBase) -> Self {
        let len = if op.is_null() { 0 } else { op.num_attributes() };
        Self(OpElementIter { op, len, pos: 0 })
    }
}
impl Iterator for OpAttrIter {
    type Item = NamedAttribute;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|pos| self.0.get_attribute(pos))
    }
}
impl FusedIterator for OpAttrIter {}

/// Iterator used for walking some set of linked operations
pub struct OperationIter {
    op: Option<OperationBase>,
}
impl OperationIter {
    #[inline]
    pub fn new(op: Option<OperationBase>) -> Self {
        Self { op }
    }
}
impl Iterator for OperationIter {
    type Item = OperationBase;

    fn next(&mut self) -> Option<Self::Item> {
        match self.op {
            None => None,
            Some(op) => {
                self.op = op.next();
                self.op
            }
        }
    }
}
impl FusedIterator for OperationIter {}

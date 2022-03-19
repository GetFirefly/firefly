use super::*;
use crate::support::StringRef;

macro_rules! impl_instruction_traits {
    ($name:ident) => {
        /// Creates a copy of this instruction which is identical except for the following:
        ///
        /// * It has no parent block
        /// * It has no name
        impl Clone for $name {
            fn clone(&self) -> Self {
                Self(unsafe { LLVMInstructionClone(self.0) })
            }
        }
        impl Instruction for $name {}
        impl Value for $name {
            fn base(&self) -> ValueBase {
                self.0
            }
        }
        impl Into<ValueBase> for $name {
            fn into(self) -> ValueBase {
                self.0
            }
        }
    };
}

/// This trait is implemented by all values that derive from llvm::Instruction
pub trait Instruction: Value {
    fn opcode(&self) -> Opcode {
        unsafe { LLVMGetInstructionOpcode(self.base()) }
    }

    fn block(&self) -> Block {
        extern "C" {
            fn LLVMGetInstructionParent(inst: ValueBase) -> Block;
        }

        unsafe { LLVMGetInstructionParent(self.base()) }
    }

    fn has_metadata(&self) -> bool {
        extern "C" {
            fn LLVMHasMetadata(inst: ValueBase) -> bool;
        }
        unsafe { LLVMHasMetadata(self.base()) }
    }

    fn get_metadata(&self, kind: MetadataKind) -> Option<MetadataValue> {
        extern "C" {
            fn LLVMGetMetadata(inst: ValueBase, kind: MetadataKind) -> MetadataValue;
        }
        let value = unsafe { LLVMGetMetadata(self.base(), kind) };
        if value.is_null() {
            None
        } else {
            Some(value)
        }
    }

    fn set_metadata(&self, kind: MetadataKind, value: MetadataValue) {
        extern "C" {
            fn LLVMSetMetadata(inst: ValueBase, kind: MetadataKind, value: MetadataValue);
        }
        unsafe { LLVMSetMetadata(self.base(), kind, value) }
    }

    /// Remove this instruction from its containing block, but keep it alive
    fn detach(&self) {
        extern "C" {
            fn LLVMInstructionRemoveFromParent(inst: ValueBase);
        }
        unsafe { LLVMInstructionRemoveFromParent(self.base()) }
    }

    /// Remove this instruction from its containing block and destroy it
    fn delete(&self) {
        extern "C" {
            fn LLVMInstructionEraseFromParent(inst: ValueBase);
        }
        unsafe { LLVMInstructionEraseFromParent(self.base()) }
    }

    /// Returns true if this instruction is a terminator instruction (e.g. BrInst)
    fn is_terminator(&self) -> bool {
        extern "C" {
            fn LLVMIsATerminatorInst(inst: ValueBase) -> bool;
        }
        unsafe { LLVMIsATerminatorInst(self.base()) }
    }

    /// Set the debug location for this instruction
    ///
    /// To clear existing debug location metadata, pass `Metadata::null()`
    fn set_debug_loc(&self, loc: Metadata) {
        extern "C" {
            fn LLVMInstructionSetDebugLoc(inst: ValueBase, loc: Metadata);
        }
        unsafe { LLVMInstructionSetDebugLoc(self.base(), loc) }
    }
}

/// This trait is implemented on all instruction types that are valid terminators
///
/// Terminators can have successors, all other instructions can't.
pub trait Terminator: Instruction {
    /// Get the number of successors this instruction has
    fn num_successors(&self) -> usize {
        extern "C" {
            fn LLVMGetNumSuccessors(inst: ValueBase) -> u32;
        }
        unsafe { LLVMGetNumSuccessors(self.base()) as usize }
    }

    /// Get the successor at the given index
    fn get_successor(&self, index: usize) -> Block {
        extern "C" {
            fn LLVMGetSuccessor(value: ValueBase, index: u32) -> Block;
        }
        debug_assert!(index < self.num_successors());
        unsafe { LLVMGetSuccessor(self.base(), index.try_into().unwrap()) }
    }

    fn set_successor(&self, index: usize, block: Block) {
        extern "C" {
            fn LLVMSetSuccessor(value: ValueBase, index: u32, block: Block);
        }
        debug_assert!(index < self.num_successors());
        unsafe { LLVMSetSuccessor(self.base(), index.try_into().unwrap(), block) }
    }
}

/// This trait is implemented on all instruction types which derive from CallInst
///
/// * intrinsics
/// * calls
/// * invokes
pub trait Call: Instruction {
    /// Returns the number of argument operands given to this call
    fn num_arguments(&self) -> usize {
        extern "C" {
            fn LLVMGetNumArgOperands(inst: ValueBase) -> u32;
        }
        unsafe { LLVMGetNumArgOperands(self.base()) as usize }
    }

    /// Returns the calling convention for this call
    fn calling_convention(&self) -> CallConv {
        extern "C" {
            fn LLVMGetInstructionCallConv(inst: ValueBase) -> u32;
        }
        unsafe { LLVMGetInstructionCallConv(self.base()) }.into()
    }

    /// Sets the calling convention of this call
    fn set_calling_convention(&self, cc: CallConv) {
        extern "C" {
            fn LLVMSetInstructionCallConv(inst: ValueBase, cc: u32);
        }
        unsafe { LLVMSetInstructionCallConv(self.base(), cc.into()) }
    }

    /// Adds the given attribute as a callsite attribute at the given location
    fn add_call_site_attr<A: Attribute>(&self, value: A, index: AttributePlace) {
        extern "C" {
            fn LLVMAddCallSiteAttribute(inst: ValueBase, index: u32, value: AttributeBase);
        }
        unsafe { LLVMAddCallSiteAttribute(self.base(), index.into(), value.base()) }
    }

    /// Get a callsite enum attribute with the given kind and location
    fn get_call_site_enum_attr(
        &self,
        index: AttributePlace,
        kind: AttributeKind,
    ) -> Option<EnumAttribute> {
        extern "C" {
            fn LLVMGetCallSiteEnumAttribute(
                inst: ValueBase,
                index: u32,
                kind: AttributeKind,
            ) -> EnumAttribute;
        }
        let attr = unsafe { LLVMGetCallSiteEnumAttribute(self.base(), index.into(), kind) };
        if attr.is_null() {
            None
        } else {
            Some(attr)
        }
    }

    /// Get a callsite string attribute with the given name and location
    fn get_call_site_string_attr<K: Into<StringRef>>(
        &self,
        index: AttributePlace,
        key: K,
    ) -> Option<StringAttribute> {
        extern "C" {
            fn LLVMGetCallSiteStringAttribute(
                inst: ValueBase,
                index: u32,
                key: *const u8,
                key_len: u32,
            ) -> StringAttribute;
        }
        let key = key.into();
        let attr = unsafe {
            LLVMGetCallSiteStringAttribute(
                self.base(),
                index.into(),
                key.data,
                key.len.try_into().unwrap(),
            )
        };
        if attr.is_null() {
            None
        } else {
            Some(attr)
        }
    }

    /// Returns the callee value given to this instruction
    ///
    /// For direct calls, you can expect this to be a function value, but for
    /// indirect calls it will be an SSA value of function pointer type.
    ///
    /// NOTE: Unknown if there are any other valid callee values.
    fn callee(&self) -> ValueBase {
        extern "C" {
            fn LLVMGetCalledValue(inst: ValueBase) -> ValueBase;
        }
        unsafe { LLVMGetCalledValue(self.base()) }
    }

    /// Returns the function type of the callee
    fn callee_type(&self) -> FunctionType {
        extern "C" {
            fn LLVMGetCalledFunctionType(inst: ValueBase) -> FunctionType;
        }
        unsafe { LLVMGetCalledFunctionType(self.base()) }
    }

    /// Returns true if this call is a tail call
    fn is_tail_call(&self) -> bool {
        extern "C" {
            fn LLVMIsTailCall(inst: ValueBase) -> bool;
        }
        unsafe { LLVMIsTailCall(self.base()) }
    }

    /// Mark this call as a tail call
    fn set_tail_call(&self, is_tail: bool) {
        extern "C" {
            fn LLVMSetTailCall(inst: ValueBase, is_tail: bool);
        }
        unsafe { LLVMSetTailCall(self.base(), is_tail) }
    }
}

/// An opaque handle to an instruction of unknown type/provenance
///
/// This is primarily used for certain operations on instructions where
/// we get the instruction value from the FFI bridge and have no need to
/// determine its actual concrete type
#[repr(transparent)]
#[derive(Copy)]
pub struct InstructionBase(ValueBase);
impl_instruction_traits!(InstructionBase);
impl TryFrom<ValueBase> for InstructionBase {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => Ok(Self(value)),
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Used to represent conditional and unconditional branches/jumps
///
/// NOTE: This does not include things like callbr or switch
#[repr(transparent)]
#[derive(Copy)]
pub struct BranchInst(ValueBase);
impl_instruction_traits!(BranchInst);
impl Terminator for BranchInst {}
impl BranchInst {
    /// Returns true if this branch has a conditional
    pub fn is_conditional(self) -> bool {
        extern "C" {
            fn LLVMIsConditional(value: BranchInst) -> bool;
        }
        unsafe { LLVMIsConditional(self) }
    }

    /// Returns the condition for this conditional branch instruction
    ///
    /// NOTE: This function will panic if this is not a conditional branch instruction
    pub fn condition(self) -> ValueBase {
        extern "C" {
            fn LLVMGetCondition(value: BranchInst) -> ValueBase;
        }
        unsafe { LLVMGetCondition(self) }
    }
}
impl TryFrom<ValueBase> for BranchInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Br | Opcode::IndirectBr => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Used to represent conditional and unconditional branches/jumps
///
/// NOTE: This does not include things like callbr or switch
#[repr(transparent)]
#[derive(Copy)]
pub struct SwitchInst(ValueBase);
impl_instruction_traits!(SwitchInst);
impl Terminator for SwitchInst {}
impl SwitchInst {
    /// Returns the default destination block for this switch instruction
    pub fn default_destination(self) -> Block {
        extern "C" {
            fn LLVMGetSwitchDefaultDest(value: SwitchInst) -> Block;
        }
        unsafe { LLVMGetSwitchDefaultDest(self) }
    }

    /// Adds a new case to this switch
    pub fn add_case<V: Value>(self, value: V, dest: Block) {
        extern "C" {
            fn LLVMAddCase(sw: SwitchInst, value: ValueBase, dest: Block);
        }
        unsafe { LLVMAddCase(self, value.base(), dest) }
    }
}
impl TryFrom<ValueBase> for SwitchInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Switch => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// A concrete instruction type for standard function calls
#[repr(transparent)]
#[derive(Copy)]
pub struct CallInst(ValueBase);
impl_instruction_traits!(CallInst);
impl Call for CallInst {}
impl TryFrom<ValueBase> for CallInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Call => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// A concrete instruction type for function calls with exception handling
#[repr(transparent)]
#[derive(Copy)]
pub struct InvokeInst(ValueBase);
impl_instruction_traits!(InvokeInst);
impl Call for InvokeInst {}
impl InvokeInst {
    /// Returns the normal destination block, i.e. where control resumes if the call does not unwind
    pub fn normal_destination(self) -> Block {
        extern "C" {
            fn LLVMGetNormalDest(invoke: InvokeInst) -> Block;
        }
        unsafe { LLVMGetNormalDest(self) }
    }

    /// Sets the normal destination for this invoke to `block`
    pub fn set_normal_destination(self, block: Block) {
        extern "C" {
            fn LLVMSetNormalDest(invoke: InvokeInst, block: Block);
        }
        unsafe {
            LLVMSetNormalDest(self, block);
        }
    }

    /// Returns the unwind destination block, i.e. where control resumes if the call unwinds
    pub fn unwind_destination(self) -> Block {
        extern "C" {
            fn LLVMGetUnwindDest(invoke: InvokeInst) -> Block;
        }
        unsafe { LLVMGetUnwindDest(self) }
    }

    /// Sets the unwind destination for this invoke to `block`
    pub fn set_unwind_destination(self, block: Block) {
        extern "C" {
            fn LLVMSetUnwindDest(invoke: InvokeInst, block: Block);
        }
        unsafe {
            LLVMSetUnwindDest(self, block);
        }
    }
}
impl TryFrom<ValueBase> for InvokeInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Invoke => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the exception handling intrinsic for resuming propagation of an in-flight exception
#[repr(transparent)]
#[derive(Copy)]
pub struct ResumeInst(ValueBase);
impl_instruction_traits!(ResumeInst);
impl Terminator for ResumeInst {}
impl TryFrom<ValueBase> for ResumeInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Resume => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the exception handling intrinsic for setting up an exception landing pad block
#[repr(transparent)]
#[derive(Copy)]
pub struct LandingPadInst(ValueBase);
impl_instruction_traits!(LandingPadInst);
impl LandingPadInst {
    /// Add a catch or filter clause to this landing pad instruction
    pub fn add_clause(self, clause: ValueBase) {
        extern "C" {
            fn LLVMAddClause(lp: LandingPadInst, clause: ValueBase);
        }
        unsafe {
            LLVMAddClause(self, clause);
        }
    }

    /// Returns true if this landingpad has the `cleanup` flag set
    pub fn is_cleanup(self) -> bool {
        extern "C" {
            fn LLVMIsCleanup(pad: LandingPadInst) -> bool;
        }
        unsafe { LLVMIsCleanup(self) }
    }

    /// If true, sets the `cleanup` flag on this landingpad
    /// If false, clears the `cleanup` flag
    pub fn set_cleanup(self, is_cleanup: bool) {
        extern "C" {
            fn LLVMSetCleanup(pad: LandingPadInst, is_cleanup: bool);
        }
        unsafe {
            LLVMSetCleanup(self, is_cleanup);
        }
    }

    /// Returns the number of clauses this landingpad instruction has
    pub fn num_clauses(self) -> usize {
        extern "C" {
            fn LLVMGetNumHandlers(pad: LandingPadInst) -> u32;
        }
        unsafe { LLVMGetNumHandlers(self) as usize }
    }

    /// Gets the clause at the given index
    pub fn get_clause(self, index: usize) -> ValueBase {
        extern "C" {
            fn LLVMGetClause(pad: LandingPadInst, index: u32) -> ValueBase;
        }
        unsafe { LLVMGetClause(self, index.try_into().unwrap()) }
    }
}
impl TryFrom<ValueBase> for LandingPadInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::LandingPad => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the SEH exception handling intrinsic for indicating to the personality function
/// that the cleanup pad it transferred control to has ended, transferring control to continue
/// or unwind out of the function
#[repr(transparent)]
#[derive(Copy)]
pub struct CleanupRetInst(ValueBase);
impl_instruction_traits!(CleanupRetInst);
impl Terminator for CleanupRetInst {}
impl TryFrom<ValueBase> for CleanupRetInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::CleanupRet => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the SEH exception handling intrinsic for ending an existing in-flight exception
/// whose unwinding was caught with a catchpad instruction. The personality function gets a chance
/// to execute code to, e.g. destroy the exception object, control then transfers to the normal
/// destination block
#[repr(transparent)]
#[derive(Copy)]
pub struct CatchRetInst(ValueBase);
impl_instruction_traits!(CatchRetInst);
impl Terminator for CatchRetInst {}
impl TryFrom<ValueBase> for CatchRetInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::CatchRet => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the SEH exception handling intrinsic for setting up a basic block as a catch handler
#[repr(transparent)]
#[derive(Copy)]
pub struct CatchPadInst(ValueBase);
impl CatchPadInst {
    /// Get the parent catchswitch instruction
    pub fn parent(self) -> Option<CatchSwitchInst> {
        extern "C" {
            fn LLVMGetParentCatchSwitch(pad: CatchPadInst) -> ValueBase;
        }
        let cs = unsafe { LLVMGetParentCatchSwitch(self) };
        if cs.is_null() {
            None
        } else {
            Some(CatchSwitchInst(cs))
        }
    }

    /// Set the parent catchswitch instruction
    pub fn set_parent(self, parent: Option<CatchSwitchInst>) {
        extern "C" {
            fn LLVMSetParentCatchSwitch(pad: CatchPadInst, parent: ValueBase);
        }
        let parent = parent.map(|cs| cs.base()).unwrap_or_else(ValueBase::null);
        unsafe {
            LLVMSetParentCatchSwitch(self, parent);
        }
    }
}
impl_instruction_traits!(CatchPadInst);
impl TryFrom<ValueBase> for CatchPadInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::CatchPad => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the SEH exception handling intrinsic for setting up a basic block as a cleanup handler
#[repr(transparent)]
#[derive(Copy)]
pub struct CleanupPadInst(ValueBase);
impl_instruction_traits!(CleanupPadInst);
impl TryFrom<ValueBase> for CleanupPadInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::CleanupPad => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the SEH exception handling intrinsic for declaring the set of possible catch handlers
/// that may be executed by the current personality function
#[repr(transparent)]
#[derive(Copy)]
pub struct CatchSwitchInst(ValueBase);
impl_instruction_traits!(CatchSwitchInst);
impl Terminator for CatchSwitchInst {}
impl CatchSwitchInst {
    /// Adds another handler to this catchswitch
    pub fn add_handler(self, pad: Block) {
        extern "C" {
            fn LLVMAddHandler(switch: CatchSwitchInst, pad: Block);
        }
        unsafe {
            LLVMAddHandler(self, pad);
        }
    }

    /// Returns the number of handlers for this catchswitch
    pub fn num_handlers(self) -> usize {
        extern "C" {
            fn LLVMGetNumHandlers(switch: CatchSwitchInst) -> u32;
        }
        unsafe { LLVMGetNumHandlers(self) as usize }
    }

    /// Returns the set of handler blocks attached to this catchswitch
    pub fn handlers(self) -> Vec<Block> {
        extern "C" {
            fn LLVMGetHandlers(cs: CatchSwitchInst, handlers: *mut Block);
        }
        let len = self.num_handlers();
        let mut handlers = Vec::with_capacity(len);
        unsafe {
            LLVMGetHandlers(self, handlers.as_mut_ptr());
            handlers.set_len(len);
        }
        handlers
    }
}
impl TryFrom<ValueBase> for CatchSwitchInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::CatchSwitch => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents the PHI instruction
///
/// PHIs are used to join the source of a value from multiple predecessors
#[repr(transparent)]
#[derive(Copy)]
pub struct PhiInst(ValueBase);
impl PhiInst {
    /// Add incoming edges to this phi node
    pub fn add_incoming(self, incoming: &[(ValueBase, Block)]) {
        extern "C" {
            fn LLVMAddIncoming(
                phi: PhiInst,
                incoming: *const ValueBase,
                preds: *const Block,
                len: u32,
            );
        }

        let num_incoming = incoming.len().try_into().unwrap();
        let values = incoming.iter().map(|(v, _)| v).copied().collect::<Vec<_>>();
        let blocks = incoming.iter().map(|(_, b)| b).copied().collect::<Vec<_>>();

        unsafe {
            LLVMAddIncoming(self, values.as_ptr(), blocks.as_ptr(), num_incoming);
        }
    }

    /// Get the number of incoming edges on this phi node
    pub fn num_incoming(self) -> usize {
        extern "C" {
            fn LLVMCountIncoming(phi: PhiInst) -> u32;
        }
        unsafe { LLVMCountIncoming(self) as usize }
    }

    /// Get the incoming value at the given index
    pub fn get_incoming(self, index: usize) -> ValueBase {
        extern "C" {
            fn LLVMGetIncomingValue(phi: PhiInst, index: u32) -> ValueBase;
        }
        unsafe { LLVMGetIncomingValue(self, index.try_into().unwrap()) }
    }

    /// Get the originating block for the incoming value at the given index
    pub fn get_incoming_block(self, index: usize) -> Block {
        extern "C" {
            fn LLVMGetIncomingBlock(phi: PhiInst, index: u32) -> Block;
        }
        unsafe { LLVMGetIncomingBlock(self, index.try_into().unwrap()) }
    }
}
impl_instruction_traits!(PhiInst);
impl TryFrom<ValueBase> for PhiInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::PHI => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents a dynamic allocation on the stack
#[repr(transparent)]
#[derive(Copy)]
pub struct AllocaInst(ValueBase);
impl_instruction_traits!(AllocaInst);
impl Align for AllocaInst {}
impl TryFrom<ValueBase> for AllocaInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Alloca => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents loading a value via a pointer
#[repr(transparent)]
#[derive(Copy)]
pub struct LoadInst(ValueBase);
impl_instruction_traits!(LoadInst);
impl Align for LoadInst {}
impl TryFrom<ValueBase> for LoadInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Load => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents storing a value via a pointer
#[repr(transparent)]
#[derive(Copy)]
pub struct StoreInst(ValueBase);
impl_instruction_traits!(StoreInst);
impl Align for StoreInst {}
impl TryFrom<ValueBase> for StoreInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Store => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents an integer comparison operator
#[repr(transparent)]
#[derive(Copy)]
pub struct ICmpInst(ValueBase);
impl_instruction_traits!(ICmpInst);
impl ICmpInst {
    /// Returns the predicate for this comparison operator
    pub fn predicate(self) -> ICmp {
        extern "C" {
            fn LLVMGetICmpPredicate(op: ICmpInst) -> ICmp;
        }
        unsafe { LLVMGetICmpPredicate(self) }
    }
}
impl TryFrom<ValueBase> for ICmpInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::ICmp => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents a return instruction, with or with a value
#[repr(transparent)]
#[derive(Copy)]
pub struct ReturnInst(ValueBase);
impl_instruction_traits!(ReturnInst);
impl Terminator for ReturnInst {}
impl TryFrom<ValueBase> for ReturnInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Ret => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents an unreachable instruction
#[repr(transparent)]
#[derive(Copy)]
pub struct UnreachableInst(ValueBase);
impl_instruction_traits!(UnreachableInst);
impl Terminator for UnreachableInst {}
impl TryFrom<ValueBase> for UnreachableInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::Unreachable => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents extraction of the value of a struct member or array element
#[repr(transparent)]
#[derive(Copy)]
pub struct ExtractValueInst(ValueBase);
impl_instruction_traits!(ExtractValueInst);
impl TryFrom<ValueBase> for ExtractValueInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::ExtractValue => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Represents insertion of the value of a struct member or array element
#[repr(transparent)]
#[derive(Copy)]
pub struct InsertValueInst(ValueBase);
impl_instruction_traits!(InsertValueInst);
impl TryFrom<ValueBase> for InsertValueInst {
    type Error = InvalidTypeCastError;

    fn try_from(value: ValueBase) -> Result<Self, Self::Error> {
        match value.kind() {
            ValueKind::Instruction => match unsafe { LLVMGetInstructionOpcode(value) } {
                Opcode::InsertValue => Ok(Self(value)),
                _ => Err(InvalidTypeCastError),
            },
            _ => Err(InvalidTypeCastError),
        }
    }
}

/// Iterator for blocks in a function
pub(super) struct InstIter(InstructionBase);
impl InstIter {
    pub(super) fn new(start: InstructionBase) -> Self {
        Self(start)
    }
}
impl Iterator for InstIter {
    type Item = InstructionBase;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMGetNextInstruction(inst: InstructionBase) -> InstructionBase;
        }
        if self.0.is_null() {
            return None;
        }
        let next = self.0;
        self.0 = unsafe { LLVMGetNextInstruction(next) };
        Some(next)
    }
}
impl std::iter::FusedIterator for InstIter {}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum Opcode {
    // Terminator Instructions
    Ret = 1,
    Br = 2,
    Switch = 3,
    IndirectBr = 4,
    Invoke = 5,
    // removed 6 due to API changes
    Unreachable = 7,
    CallBr = 67,

    // Standard Unary Operators
    FNeg = 66,

    // Standard Binary Operators
    Add = 8,
    FAdd = 9,
    Sub = 10,
    FSub = 11,
    Mul = 12,
    FMul = 13,
    UDiv = 14,
    SDiv = 15,
    FDiv = 16,
    URem = 17,
    SRem = 18,
    FRem = 19,

    // Logical Operators
    Shl = 20,
    LShr = 21,
    AShr = 22,
    And = 23,
    Or = 24,
    Xor = 25,

    // Memory Operators
    Alloca = 26,
    Load = 27,
    Store = 28,
    GetElementPtr = 29,

    // Cast Operators
    Trunc = 30,
    ZExt = 31,
    SExt = 32,
    FPToUI = 33,
    FPToSI = 34,
    UIToFP = 35,
    SIToFP = 36,
    FPTrunc = 37,
    FPExt = 38,
    PtrToInt = 39,
    IntToPtr = 40,
    BitCast = 41,
    AddrSpaceCast = 60,

    // Other Operators
    ICmp = 42,
    FCmp = 43,
    PHI = 44,
    Call = 45,
    Select = 46,
    UserOp1 = 47,
    UserOp2 = 48,
    VAArg = 49,
    ExtractElement = 50,
    InsertElement = 51,
    ShuffleVector = 52,
    ExtractValue = 53,
    InsertValue = 54,
    Freeze = 68,

    // Atomic operators
    Fence = 55,
    AtomicCmpXchg = 56,
    AtomicRMW = 57,

    // Exception Handling Operators
    Resume = 58,
    LandingPad = 59,
    CleanupRet = 61,
    CatchRet = 62,
    CatchPad = 63,
    CleanupPad = 64,
    CatchSwitch = 65,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum ICmp {
    Eq = 32,
    Ne,
    Ugt,
    Uge,
    Ult,
    Ule,
    Sgt,
    Sge,
    Slt,
    Sle,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum FCmp {
    /// Always false (and folded)
    False = 0,
    OrderedEq,
    OrderedGt,
    OrderedGe,
    OrderedLt,
    OrderedLe,
    OrderedNe,
    /// True if ordered (i.e. no nans)
    Ordered,
    /// True if unordered (i.e. either operand is nan)
    Unordered,
    UnorderedEq,
    UnorderedGt,
    UnorderedGe,
    UnorderedLt,
    UnorderedLe,
    UnorderedNe,
    /// Always true (and folded)
    True,
}

extern "C" {
    fn LLVMGetInstructionOpcode(inst: ValueBase) -> Opcode;
    fn LLVMInstructionClone(inst: ValueBase) -> ValueBase;
}

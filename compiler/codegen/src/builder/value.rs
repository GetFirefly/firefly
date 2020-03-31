use cranelift_entity::entity_impl;
use cranelift_entity::packed_option::{PackedOption, ReservedValue};

use libeir_ir as ir;

use liblumen_mlir::ir::ValueRef;

/// An opaque reference to an SSA value.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Value(u32);
entity_impl!(Value, "mlir_value");
impl Default for Value {
    fn default() -> Self {
        Value::reserved_value()
    }
}

/// Metadata about an SSA value
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValueData {
    /// A reference to the value in MLIR
    pub value: ValueRef,

    /// The original EIR value this value was created in reference to
    /// NOTE: Not all values present in MLIR may have a source in EIR
    pub ir_value: PackedOption<ir::Value>,

    /// Contains metadata about the value (its purpose, type, etc.)
    pub data: ValueDef,
}

/// Metadata about a value's definition
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueDef {
    /// Value is the n'th result of an instruction.
    Result(usize),
    /// Value is the n'th parameter to a block
    Param(usize),
    /// Value is the n'th value in the source closure environment
    Env(usize),
    /// Value is the return continuation
    Return,
    /// Value is the escape continuation
    Escape,
}

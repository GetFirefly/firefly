use cranelift_entity::entity_impl;
use cranelift_entity::packed_option::{PackedOption, ReservedValue};

use liblumen_mlir::ir::BlockRef;

use super::value::Value;
use crate::builder::function::Param;

/// An opaque reference to a block in a function
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Block(u32);
entity_impl!(Block, "mlir_block");
impl Default for Block {
    fn default() -> Self {
        Block::reserved_value()
    }
}

pub struct BlockData {
    // The pointer to the MLIR block this represents
    pub block: BlockRef,
    pub ir_block: PackedOption<libeir_ir::Block>,
    pub params: Vec<Param>,
    pub param_values: Vec<Value>,
}

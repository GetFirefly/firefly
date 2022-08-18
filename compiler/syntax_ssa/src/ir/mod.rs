pub mod block;
pub mod builders;
pub mod constants;
pub mod dataflow;
pub mod function;
pub mod instructions;
pub mod layout;
pub mod module;
pub mod value;

pub use self::block::{Block, BlockData};
pub use self::builders::{InstBuilder, InstBuilderBase};
pub use self::constants::{
    Constant, ConstantData, ConstantItem, ConstantPool, Immediate, ImmediateTerm, IntoBytes,
};
pub use self::dataflow::DataFlowGraph;
pub use self::function::{FuncRef, Function};
pub use self::instructions::*;
pub use self::layout::{ArenaMap, LayoutAdapter, LayoutNode, OrderedArenaMap};
pub use self::module::Module;
pub use self::value::{Value, ValueData, ValueList, ValueListPool, Values};

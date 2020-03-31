mod ffi {
    use liblumen_compiler_macros::foreign_struct;

    #[foreign_struct]
    pub struct Attribute;
    #[foreign_struct]
    pub struct Block;
    #[foreign_struct]
    pub struct FunctionOp;
    #[foreign_struct]
    pub struct Location;
    #[foreign_struct]
    pub struct Value;
}

pub use self::ffi::{AttributeRef, BlockRef, FunctionOpRef, LocationRef, ValueRef};

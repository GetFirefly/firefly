pub mod ops;

use liblumen_mlir::*;

pub struct LLVMBuilder<'a> {
    builder: &'a mut OpBuilder,
}

extern "C" {}

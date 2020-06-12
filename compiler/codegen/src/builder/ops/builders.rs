use anyhow::anyhow;

use log::debug;

use libeir_ir as ir;

use liblumen_mlir::ir::*;

use crate::builder::block::Block;
use crate::builder::ffi::*;
use crate::builder::function::ScopedFunctionBuilder;
use crate::builder::ops::*;
use crate::builder::value::{Value, ValueDef};
use crate::Result;

mod binary;
mod binary_operators;
mod call;
mod closures;
mod constants;
mod control;
mod intrinsics;
mod list;
mod logical_operators;
mod map;
mod patterns;
mod receive;
mod trace;
mod tuple;

pub use self::binary::*;
pub use self::binary_operators::*;
pub use self::call::*;
pub use self::closures::*;
pub use self::constants::*;
pub use self::control::*;
pub use self::intrinsics::*;
pub use self::list::*;
pub use self::logical_operators::*;
pub use self::map::*;
pub use self::patterns::*;
pub use self::receive::*;
pub use self::trace::*;
pub use self::tuple::*;

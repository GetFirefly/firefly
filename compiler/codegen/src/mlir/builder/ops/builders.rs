use anyhow::anyhow;

use log::debug;

use crate::mlir::builder::block::Block;
use crate::mlir::builder::ffi::{self, *};
use crate::mlir::builder::function::ScopedFunctionBuilder;
use crate::mlir::builder::ops::{self, *};
use crate::mlir::builder::value::{Value, ValueDef};
use crate::Result;

use super::builder::OpBuilder;

use libeir_intern::Symbol;
use libeir_ir as ir;

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
pub use self::trace::*;
pub use self::tuple::*;

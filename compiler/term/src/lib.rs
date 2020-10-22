#![feature(arbitrary_enum_discriminant)]
#![feature(unwind_attributes)]

mod encoding;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod tag;

pub use self::encoding::*;
#[cfg(not(target_arch = "wasm32"))]
pub use self::ffi::{TermKind, Type};
pub use self::tag::Tag;

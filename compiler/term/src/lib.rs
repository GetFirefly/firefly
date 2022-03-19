#![feature(arbitrary_enum_discriminant)]
#![feature(c_unwind)]

mod encoding;
#[cfg(not(target_arch = "wasm32"))]
mod ffi;
mod tag;

pub use self::encoding::*;
#[cfg(not(target_arch = "wasm32"))]
pub use self::ffi::TermKind;
pub use self::tag::Tag;

#![deny(warnings)]
#![feature(trait_alias)]
#![feature(map_try_insert)]
#![feature(iterator_try_collect)]
#![feature(box_patterns)]
#![feature(once_cell)]

#[macro_use]
mod macros;
mod ast;
mod evaluator;
pub mod features;
mod lexer;
mod parser;
pub mod passes;
mod preprocessor;
mod visit;

pub use self::ast::*;
pub use self::lexer::*;
pub use self::parser::*;
pub use self::preprocessor::*;

use std::sync::OnceLock;

pub static OTP_RELEASE: OnceLock<usize> = OnceLock::new();

/// Returns the OTP_RELEASE value for the current build
pub fn otp_release() -> usize {
    let release = OTP_RELEASE.get_or_init(|| {
        option_env!("OTP_RELEASE")
            .unwrap_or("25")
            .parse()
            .expect("invalid OTP_RELEASE value")
    });
    *release
}

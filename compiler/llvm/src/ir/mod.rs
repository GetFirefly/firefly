mod attributes;
mod block;
mod comdat;
mod constants;
mod context;
mod funclet;
mod function;
mod globals;
mod instructions;
mod metadata;
mod module;
mod types;
mod users;
mod value;

pub use self::attributes::*;
pub use self::block::*;
pub use self::comdat::*;
pub use self::constants::*;
pub use self::context::*;
pub use self::funclet::*;
pub use self::function::*;
pub use self::globals::*;
pub use self::instructions::*;
pub use self::metadata::*;
pub use self::module::*;
pub use self::types::*;
pub use self::users::*;
pub use self::value::*;

/// This error is used when performing casts/conversions
/// between IR types of the same category
#[derive(Debug, thiserror::Error)]
#[error("invalid cast between types")]
pub struct InvalidTypeCastError;

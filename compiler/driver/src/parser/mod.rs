mod queries;
mod query_groups;

pub use self::query_groups::{Parser, ParserStorage};

pub(crate) mod prelude {
    pub use super::query_groups::{Parser, ParserStorage};
    pub use crate::diagnostics::*;
    pub use crate::interner::*;
    pub use crate::output::CompilerOutput;
}

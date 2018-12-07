mod expect;
mod parse;
mod preprocessor;
mod token_read;

use crate::syntax::tokenizer::values::Symbol;

pub use self::expect::Expect;
pub use self::parse::{Parse, ParseTail};
pub use self::preprocessor::Preprocessor;
pub use self::token_read::TokenRead;

pub trait Delimiter {
    fn delimiter() -> Symbol;
}

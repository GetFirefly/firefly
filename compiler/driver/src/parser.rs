mod queries;

use std::path::PathBuf;
use std::sync::Arc;

use liblumen_session::{IRModule, InputType, Options};
use liblumen_util::seq::Seq;

use libeir_syntax_erl::ParseConfig;

pub(crate) mod prelude {
    pub use super::Parser;
    pub use crate::diagnostics::*;
    pub use crate::interner::{InternedInput, Interner};
    pub use crate::output::CompilerOutput;
}

use self::prelude::*;

#[salsa::query_group(ParserStorage)]
pub trait Parser: CompilerOutput {
    #[salsa::input]
    fn options(&self) -> Arc<Options>;

    #[salsa::invoke(queries::output_dir)]
    fn output_dir(&self) -> PathBuf;

    #[salsa::invoke(queries::inputs)]
    fn inputs(&self) -> QueryResult<Arc<Seq<InternedInput>>>;

    #[salsa::invoke(queries::input_type)]
    fn input_type(&self, input: InternedInput) -> InputType;

    #[salsa::invoke(queries::parse_config)]
    fn parse_config(&self) -> ParseConfig;

    #[salsa::invoke(queries::input_parsed)]
    fn input_parsed(&self, input: InternedInput) -> QueryResult<IRModule>;

    #[salsa::invoke(queries::input_eir)]
    fn input_eir(&self, input: InternedInput) -> QueryResult<IRModule>;
}

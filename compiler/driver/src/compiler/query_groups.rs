use std::sync::Arc;
use std::thread::ThreadId;

use firefly_codegen::meta::CompiledModule;
use firefly_syntax_base::ApplicationMetadata;

use crate::compiler::queries;
use crate::diagnostics::ErrorReported;
use crate::interner::*;
use crate::parser::Parser;

#[salsa::query_group(CompilerStorage)]
pub trait Compiler: Parser {
    #[salsa::invoke(queries::compile)]
    fn compile(
        &self,
        thread_id: ThreadId,
        input: InternedInput,
        app: Arc<ApplicationMetadata>,
    ) -> Result<Option<CompiledModule>, ErrorReported>;
}

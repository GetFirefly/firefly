use std::sync::Arc;

use liblumen_util::diagnostics::{CodeMap, Diagnostic, DiagnosticsHandler};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorReported;

pub type QueryResult<T> = std::result::Result<T, ErrorReported>;

pub trait CompilerDiagnostics {
    fn diagnostics(&self) -> &Arc<DiagnosticsHandler>;

    #[inline]
    fn diagnostic(&self, diag: &Diagnostic) {
        self.diagnostics().emit(diag);
    }

    #[inline]
    fn to_query_result<T>(&self, err: anyhow::Result<T>) -> QueryResult<T> {
        match err {
            Ok(val) => Ok(val),
            Err(err) => {
                self.report_error(format!("{}", err));
                Err(ErrorReported)
            }
        }
    }

    #[inline]
    fn report_error(&self, err: impl Into<String>) {
        self.diagnostics().error(err);
    }

    fn codemap(&self) -> &Arc<CodeMap>;
}

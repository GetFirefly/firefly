use std::sync::Arc;

use firefly_util::diagnostics::{CodeMap, Diagnostic, DiagnosticsHandler};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErrorReported;

pub trait CompilerDiagnostics {
    fn diagnostics(&self) -> &Arc<DiagnosticsHandler>;

    #[inline]
    fn diagnostic(&self, diag: &Diagnostic) {
        self.diagnostics().emit(diag);
    }

    #[inline]
    fn to_query_result<T>(&self, err: anyhow::Result<T>) -> Result<T, ErrorReported> {
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

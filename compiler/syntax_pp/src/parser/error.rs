use liblumen_diagnostics::*;

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("error occurred when reading {path:?}: {source}")]
    RootFileError {
        #[source]
        source: std::io::Error,
        path: std::path::PathBuf,
    },
    #[error("parsing failed: {0}")]
    LalrPop(#[from] lalrpop_util::ParseError<SourceIndex, Token, ()>),
}
impl ToDiagnostic for ParseError {
    fn to_diagnostic(&self) -> Diagnostic {
        use lalrpop_util::ParseError::*;
        match self {
            Self::RootFileError { source, path } => Diagnostic::error().with_message(format!(
                "error occurred while reading {:?}: {}",
                path, source
            )),
            Self::LalrPop(InvalidToken { location }) => {
                let source_id = location.source_id();
                let index = *location;
                Diagnostic::error()
                    .with_message("invalid token")
                    .with_labels(vec![Label::primary(
                        source_id,
                        SourceSpan::new(index, index),
                    )
                    .with_message("invalid token encountered here")])
            }
            Self::LalrPop(UnrecognizedEOF { location, expected }) => {
                let source_id = location.source_id();
                let index = *location;
                Diagnostic::error()
                    .with_message("unexpected end of file")
                    .with_labels(vec![Label::primary(
                        source_id,
                        SourceSpan::new(index, index),
                    )
                    .with_message(&format!("expected one of: {}", expected.join(", ")))])
            }
            Self::LalrPop(ExtraToken { token: (l, _, r) }) => Diagnostic::error()
                .with_message("extra token")
                .with_labels(vec![Label::primary(l.source_id(), SourceSpan::new(*l, *r))
                    .with_message("did not expect this token")]),
            Self::LalrPop(UnrecognizedToken {
                token: (l, _, r), ..
            }) => Diagnostic::error()
                .with_message("unexpected token")
                .with_labels(vec![Label::primary(l.source_id(), SourceSpan::new(*l, *r))
                    .with_message("did not expect this token")]),
            Self::LalrPop(User { .. }) => Diagnostic::error().with_message("parsing failed"),
        }
    }
}

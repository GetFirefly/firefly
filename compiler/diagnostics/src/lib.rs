mod codemap;
mod filename;
mod index;
mod source;
mod span;

pub use codespan::Location;
pub use codespan::{ByteIndex, ByteOffset};
pub use codespan::{ColumnIndex, ColumnNumber, ColumnOffset};
pub use codespan::{Index, Offset};
pub use codespan::{LineIndex, LineNumber, LineOffset};
pub use codespan::{RawIndex, RawOffset};

pub use codespan_reporting::diagnostic::{LabelStyle, Severity};
pub use codespan_reporting::files::{Error, Files};
pub use codespan_reporting::term;

pub use firefly_diagnostics_macros::*;

pub use self::codemap::CodeMap;
pub use self::filename::FileName;
pub use self::index::SourceIndex;
pub use self::source::{SourceFile, SourceId};
pub use self::span::{SourceSpan, Span, Spanned};

pub type Diagnostic = codespan_reporting::diagnostic::Diagnostic<SourceId>;
pub type Label = codespan_reporting::diagnostic::Label<SourceId>;

pub trait ToDiagnostic {
    fn to_diagnostic(self) -> Diagnostic;
}
impl ToDiagnostic for Diagnostic {
    #[inline(always)]
    fn to_diagnostic(self) -> Diagnostic {
        self
    }
}

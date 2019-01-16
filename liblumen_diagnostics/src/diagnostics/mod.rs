//! This is a forked version of `codespan` and `codespan-reporting` vendored into this project:
//!
//! * [codespan](https://github.com/brendanzab/codespan)
//!
//! Utilities for working with source code and printing nicely formatted
//! diagnostic information like warnings and errors.
mod codemap;
mod filemap;
mod index;
mod span;

pub use self::codemap::CodeMap;
pub use self::filemap::{ByteIndexError, LineIndexError, LocationError, SpanError};
pub use self::filemap::{FileMap, FileName};
pub use self::index::{ByteIndex, ByteOffset};
pub use self::index::{ColumnIndex, ColumnNumber, ColumnOffset};
pub use self::index::{Index, Offset};
pub use self::index::{LineIndex, LineNumber, LineOffset};
pub use self::index::{RawIndex, RawOffset};
pub use self::span::{ByteSpan, Span, DUMMY_SPAN};

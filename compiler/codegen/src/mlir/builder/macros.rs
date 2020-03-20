#[macro_export]
macro_rules! with_encoding {
    ($encoding:expr, $width:expr, $body:expr) => {{
        use liblumen_term::EncodingType;
        match $encoding {
            EncodingType::Encoding32 | EncodingType::Default if $width == 32 => {
                use liblumen_term::{Encoding as TermEncoding, Encoding32 as Encoding};
                $body
            }
            EncodingType::Encoding64 | EncodingType::Default if $width == 64 => {
                use liblumen_term::{Encoding as TermEncoding, Encoding64 as Encoding};
                $body
            }
            EncodingType::Encoding64Nanboxed => {
                use liblumen_term::{Encoding as TermEncoding, Encoding64Nanboxed as Encoding};
                $body
            }
            kind => {
                panic!(
                    "invalid encoding type {:#?} for target pointer width of {}",
                    kind, $width
                );
            }
        }
    }};
}

macro_rules! unwrap_const_atom {
    ($function:expr, $c:expr) => {{
        use libeir_ir::*;
        match $function.const_kind($c) {
            ConstKind::Atomic(AtomicTerm::Atom(AtomTerm(symbol))) => *symbol,
            other => panic!("expected atom constant, but got {:#?}", other),
        }
    }};
}

macro_rules! unwrap_const_int {
    ($function:expr, $c:expr) => {{
        use libeir_ir::*;
        match $function.const_kind($c) {
            ConstKind::Atomic(AtomicTerm::Int(IntTerm(i))) => *i,
            other => panic!("expected integer constant, but got {:#?}", other),
        }
    }};
}

/// Used with ScopedFunctionBuilder, but placing it here makes it available
/// to all of the sub-builders
#[macro_export]
macro_rules! debug_in {
    ($this:expr, $format:expr) => {
        debug!("{}: {}", $this.name(), $format);
    };
    ($this:expr, $format:expr, $($arg:expr),+) => {
        debug!("{}: {}", $this.name(), &format!($format, $($arg),+));
    }
}

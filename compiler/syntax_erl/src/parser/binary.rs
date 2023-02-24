use firefly_binary::{BinaryEntrySpecifier, Endianness};
use firefly_diagnostics::{Diagnostic, Label, SourceSpan, Spanned, ToDiagnostic};
use firefly_intern::{symbols, Symbol};

use crate::ast::BitType;

#[derive(Debug, thiserror::Error)]
pub enum SpecifierError {
    #[error("unknown specifier in binary entry")]
    BinaryUnknownSpecifier { span: SourceSpan },
    #[error("conflicting specifiers in binary entry")]
    BinaryConflictingSpecifier { new: SourceSpan, old: SourceSpan },
    #[error("invalid specifier for {typ} in binary entry")]
    BinaryInvalidSpecifier { span: SourceSpan, typ: Symbol },
    #[error("size is not allowed for {typ}")]
    BinarySizeNotAllowed { span: SourceSpan, typ: Symbol },
}

impl ToDiagnostic for SpecifierError {
    fn to_diagnostic(self) -> Diagnostic {
        let msg = self.to_string();
        match self {
            SpecifierError::BinaryUnknownSpecifier { span } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![
                    Label::primary(span.source_id(), span).with_message("specifier is not known")
                ]),
            SpecifierError::BinaryConflictingSpecifier { new, old } => {
                Diagnostic::error().with_message(msg).with_labels(vec![
                    Label::primary(new.source_id(), new).with_message("specifier 1"),
                    Label::primary(old.source_id(), old).with_message("specifier 2"),
                ])
            }
            SpecifierError::BinaryInvalidSpecifier { span, typ } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message(format!("specifier is not valid for {} entries", typ))]),
            SpecifierError::BinarySizeNotAllowed { span, typ } => Diagnostic::error()
                .with_message(msg)
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message(format!("size is not allowed for {} entries", typ))]),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum TypeName {
    Integer,
    Float,
    Bytes,
    Bits,
    Utf8,
    Utf16,
    Utf32,
}
impl Into<Symbol> for TypeName {
    fn into(self) -> Symbol {
        match self {
            Self::Integer => symbols::Integer,
            Self::Float => symbols::Float,
            Self::Bytes => symbols::Bytes,
            Self::Bits => symbols::Bits,
            Self::Utf8 => symbols::Utf8,
            Self::Utf16 => symbols::Utf16,
            Self::Utf32 => symbols::Utf32,
        }
    }
}

macro_rules! try_specifier {
    ($field:expr, $entry:expr, $spec:expr) => {{
        let spec = $spec;
        match $field {
            Some((f, _)) if f == spec => (),
            Some((_, old)) => {
                return Err(SpecifierError::BinaryConflictingSpecifier {
                    old,
                    new: $entry.span(),
                });
            }
            None => $field = Some((spec, $entry.span())),
        }
    }};
}

macro_rules! test_none {
    ($field:expr, $typ:expr) => {{
        if let Some((_, span)) = $field {
            return Err(SpecifierError::BinaryInvalidSpecifier {
                span,
                typ: ($typ).into(),
            });
        }
    }};
}

pub fn specifier_from_parsed(
    parsed: &[BitType],
    has_size: bool,
) -> Result<BinaryEntrySpecifier, SpecifierError> {
    let mut raw_typ = None;
    let mut signed = None;
    let mut endianness = None;
    let mut unit = None;

    for entry in parsed {
        match entry {
            // Types
            BitType::Name(_span, ident) if ident.as_str() == "integer" => {
                try_specifier!(raw_typ, entry, TypeName::Integer)
            }
            BitType::Name(_span, ident) if ident.as_str() == "float" => {
                try_specifier!(raw_typ, entry, TypeName::Float)
            }
            BitType::Name(_span, ident) if ident.as_str() == "binary" => {
                try_specifier!(raw_typ, entry, TypeName::Bytes)
            }
            BitType::Name(_span, ident) if ident.as_str() == "bytes" => {
                try_specifier!(raw_typ, entry, TypeName::Bytes)
            }
            BitType::Name(_span, ident) if ident.as_str() == "bitstring" => {
                try_specifier!(raw_typ, entry, TypeName::Bits)
            }
            BitType::Name(_span, ident) if ident.as_str() == "bits" => {
                try_specifier!(raw_typ, entry, TypeName::Bits)
            }
            BitType::Name(_span, ident) if ident.as_str() == "utf8" => {
                try_specifier!(raw_typ, entry, TypeName::Utf8)
            }
            BitType::Name(_span, ident) if ident.as_str() == "utf16" => {
                try_specifier!(raw_typ, entry, TypeName::Utf16)
            }
            BitType::Name(_span, ident) if ident.as_str() == "utf32" => {
                try_specifier!(raw_typ, entry, TypeName::Utf32)
            }

            // Signed
            BitType::Name(_span, ident) if ident.as_str() == "signed" => {
                try_specifier!(signed, entry, true)
            }
            BitType::Name(_span, ident) if ident.as_str() == "unsigned" => {
                try_specifier!(signed, entry, false)
            }

            // Endianness
            BitType::Name(_span, ident) if ident.as_str() == "big" => {
                try_specifier!(endianness, entry, Endianness::Big)
            }
            BitType::Name(_span, ident) if ident.as_str() == "little" => {
                try_specifier!(endianness, entry, Endianness::Little)
            }
            BitType::Name(_span, ident) if ident.as_str() == "native" => {
                try_specifier!(endianness, entry, Endianness::Native)
            }

            // Unit
            BitType::Sized(_span, ident, num) if ident.as_str() == "unit" => {
                try_specifier!(unit, entry, *num)
            }

            entry => {
                return Err(SpecifierError::BinaryUnknownSpecifier { span: entry.span() });
            }
        }
    }

    let typ = raw_typ.map(|(t, _)| t).unwrap_or(TypeName::Integer);

    let size_not_allowed_err = || {
        Err(SpecifierError::BinarySizeNotAllowed {
            typ: typ.into(),
            span: raw_typ.unwrap().1,
        })
    };

    let spec = match typ {
        TypeName::Integer => {
            // Default is signed-big-unit:1
            let signed = signed.map(|(t, _)| t).unwrap_or(false);
            let endianness = endianness.map(|(t, _)| t).unwrap_or(Endianness::Big);
            let unit = unit
                .map(|(t, _)| t)
                .unwrap_or(1)
                .try_into()
                .expect("invalid unit, too large");

            BinaryEntrySpecifier::Integer {
                signed,
                endianness,
                unit,
            }
        }
        TypeName::Float => {
            // Default is big-unit:1
            test_none!(signed, typ);
            let endianness = endianness.map(|(t, _)| t).unwrap_or(Endianness::Big);
            let unit = unit
                .map(|(t, _)| t)
                .unwrap_or(1)
                .try_into()
                .expect("invalid unit, too large");

            BinaryEntrySpecifier::Float { endianness, unit }
        }
        TypeName::Bytes => {
            // Default is unit:8
            test_none!(signed, typ);
            test_none!(endianness, typ);
            let unit = unit
                .map(|(t, _)| t)
                .unwrap_or(8)
                .try_into()
                .expect("invalid unit, too large");

            BinaryEntrySpecifier::Binary { unit }
        }
        TypeName::Bits => {
            // Default is unit:1
            test_none!(signed, typ);
            test_none!(endianness, typ);
            let unit = unit
                .map(|(t, _)| t)
                .unwrap_or(1)
                .try_into()
                .expect("invalid unit, too large");

            BinaryEntrySpecifier::Binary { unit }
        }
        TypeName::Utf8 => {
            test_none!(signed, typ);
            test_none!(endianness, typ);
            test_none!(unit, typ);

            if has_size {
                return size_not_allowed_err();
            }

            BinaryEntrySpecifier::Utf8
        }
        TypeName::Utf16 => {
            test_none!(signed, typ);
            let endianness = endianness.map(|(t, _)| t).unwrap_or(Endianness::Big);
            test_none!(unit, typ);

            if has_size {
                return size_not_allowed_err();
            }

            BinaryEntrySpecifier::Utf16 { endianness }
        }
        TypeName::Utf32 => {
            test_none!(signed, typ);
            let endianness = endianness.map(|(t, _)| t).unwrap_or(Endianness::Big);
            test_none!(unit, typ);

            if has_size {
                return size_not_allowed_err();
            }

            BinaryEntrySpecifier::Utf32 { endianness }
        }
    };

    Ok(spec)
}

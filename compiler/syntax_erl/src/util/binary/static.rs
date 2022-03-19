#![allow(dead_code, unused_variables)]
use liblumen_binary::{BinaryEntrySpecifier, BitVec, Endianness};
use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Ident;
use liblumen_number::{Integer, Number};

use crate::ast::{BinaryElement, Expr, Literal};
use crate::evaluator::{eval_expr, EvalError as SrcEvalError, ResolveRecordIndexError, Term};
use crate::util::encoding;
use crate::util::string_tokenizer::StringTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum StaticBinaryElementError {
    #[error(transparent)]
    EvalError {
        #[from]
        source: SrcEvalError,
    },
    #[error("size expression evaluated to invalid type, must be integer")]
    InvalidSizeType { span: SourceSpan },
    #[error("size expression evaluated to outside of valid range")]
    InvalidSizeValue { span: SourceSpan },
    #[error("native endianness not allowed in static binary expressions")]
    NativeEndianNotAllowed { span: SourceSpan },
    #[error("specifier was invalid for given value")]
    InvalidSpecifier { span: SourceSpan },
    #[error("provided value not compatible with binary element specifier")]
    IncompatibleValue { span: SourceSpan },
    #[error("only total sizes 32 and 64 are permitted for floats, got {size}")]
    InvalidFloatSize { span: SourceSpan, size: usize },
}

fn append_number(
    num: &Number,
    span: SourceSpan,
    size: Option<usize>,
    specifier: BinaryEntrySpecifier,
    out: &mut BitVec,
) -> Result<(), StaticBinaryElementError> {
    use BinaryEntrySpecifier as BES;

    match specifier {
        BES::Integer {
            endianness, unit, ..
        } => {
            let size = size.unwrap_or(8);
            let unit: usize = unit.try_into().unwrap();
            let total_size = size * unit;

            if let Number::Integer(integer) = num {
                let bitslice = integer.encode_bitstring(total_size, endianness);
                out.push(&bitslice);
                Ok(())
            } else {
                Err(StaticBinaryElementError::IncompatibleValue { span })
            }
        }
        BES::Float { unit, .. } => {
            let size = size.unwrap_or(64);
            let unit: usize = unit.try_into().unwrap();
            let total_size = size * unit;

            let float = num.to_efloat().unwrap().inner();

            match total_size {
                32 => {
                    let float = float as f32;
                    todo!()
                }
                64 => {
                    todo!()
                }
                _ => Err(StaticBinaryElementError::InvalidFloatSize {
                    span,
                    size: total_size,
                }),
            }
        }
        BES::Utf8 => {
            if let Number::Integer(integer) = num {
                let cp = integer.to_u64().unwrap();
                let encoded = encoding::encode_utf8(cp, span).unwrap();
                out.push(encoded);

                Ok(())
            } else {
                Err(StaticBinaryElementError::IncompatibleValue { span })
            }
        }
        BES::Utf16 { endianness } => {
            if let Number::Integer(integer) = num {
                let cp = integer.to_u64().unwrap();
                let mut encoded = encoding::encode_utf16(cp, span).unwrap();

                if endianness == Endianness::Little {
                    encoded = encoded.swap();
                }

                out.push(encoded);

                Ok(())
            } else {
                Err(StaticBinaryElementError::IncompatibleValue { span })
            }
        }
        BES::Utf32 { endianness } => {
            if let Number::Integer(integer) = num {
                let cp = integer.to_u64().unwrap();
                let mut encoded = encoding::encode_utf32(cp, span).unwrap();

                if endianness == Endianness::Little {
                    encoded = encoded.swap();
                }

                out.push(encoded);

                Ok(())
            } else {
                Err(StaticBinaryElementError::IncompatibleValue { span })
            }
        }
        BES::Bytes { .. } => {
            unreachable!()
        }
        BES::Bits { .. } => {
            unreachable!()
        }
    }
}

pub fn append_static_binary_element(
    elem: &BinaryElement,
    out: &mut BitVec,
    resolve_record_index: Option<&dyn Fn(Ident, Ident) -> Result<usize, ResolveRecordIndexError>>,
) -> Result<(), StaticBinaryElementError> {
    let specifier = elem.specifier.unwrap_or_else(BinaryEntrySpecifier::default);

    if specifier.is_native_endian() {
        return Err(StaticBinaryElementError::NativeEndianNotAllowed {
            span: elem.bit_expr.span(),
        });
    }

    // Evaluate bit size
    let size: Option<usize> = if let Some(size_expr) = &elem.bit_size {
        let result =
            eval_expr(size_expr, resolve_record_index).map_err(StaticBinaryElementError::from)?;
        match result {
            Term::Number(Number::Integer(num)) => {
                if let Some(integer) = num.to_usize() {
                    Some(integer)
                } else {
                    return Err(StaticBinaryElementError::InvalidSizeValue {
                        span: size_expr.span(),
                    });
                }
            }
            _ => {
                return Err(StaticBinaryElementError::InvalidSizeType {
                    span: size_expr.span(),
                });
            }
        }
    } else {
        None
    };

    // TODO switch to evaluator?
    match &elem.bit_expr {
        Expr::Literal(Literal::String(string)) => {
            let tokenizer = StringTokenizer::new(*string);
            for result in tokenizer {
                // TODO error
                let (cp, span) = result.unwrap();

                let cp_int: Integer = cp.into();
                append_number(&cp_int.into(), span, size, specifier, out)?;
            }
            Ok(())
        }
        _ => {
            let evaluated = crate::evaluator::eval_expr(&elem.bit_expr, resolve_record_index)
                .map_err(StaticBinaryElementError::from)?;

            match evaluated {
                Term::Number(number) => {
                    append_number(&number, elem.bit_expr.span(), size, specifier, out)?;
                    Ok(())
                }
                _ => Err(StaticBinaryElementError::IncompatibleValue {
                    span: elem.bit_expr.span(),
                }),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use liblumen_binary::{BinaryEntrySpecifier, BitVec, Endianness};
    use liblumen_diagnostics::SourceSpan;

    use super::*;
    use crate::ast;

    #[test]
    fn simple_integer() {
        let expr = ast::Expr::Literal(Literal::Integer(SourceSpan::UNKNOWN, 21.into()));
        let elem = ast::BinaryElement {
            span: SourceSpan::UNKNOWN,
            bit_expr: expr,
            bit_size: None,
            specifier: None,
        };

        let mut bits = BitVec::new();
        append_static_binary_element(&elem, &mut bits, None).unwrap();
        assert_eq!(bits.as_ref(), &[21]);
    }

    #[test]
    fn simple_integer_with_size() {
        let expr = ast::Expr::Literal(Literal::Integer(SourceSpan::UNKNOWN, 21.into()));
        let size_expr = ast::Expr::Literal(Literal::Integer(SourceSpan::UNKNOWN, 12.into()));
        let elem = ast::BinaryElement {
            span: SourceSpan::UNKNOWN,
            bit_expr: expr,
            bit_size: Some(size_expr),
            specifier: None,
        };

        let mut bits = BitVec::new();
        append_static_binary_element(&elem, &mut bits, None).unwrap();
        assert_eq!(bits.as_ref(), &[21 >> 4, 21 << 4]);
    }

    #[test]
    fn simple_integer_little_endian_with_size() {
        let expr = ast::Expr::Literal(Literal::Integer(SourceSpan::UNKNOWN, 21.into()));
        let size_expr = ast::Expr::Literal(Literal::Integer(SourceSpan::UNKNOWN, 12.into()));
        let elem = ast::BinaryElement {
            span: SourceSpan::UNKNOWN,
            bit_expr: expr,
            bit_size: Some(size_expr),
            specifier: Some(BinaryEntrySpecifier::Integer {
                signed: true,
                unit: 1,
                endianness: Endianness::Little,
            }),
        };

        let mut bits = BitVec::new();
        append_static_binary_element(&elem, &mut bits, None).unwrap();
        assert_eq!(bits.as_ref(), &[21, 0]);
    }
}

use std::borrow::Borrow;
use std::cmp;
use std::fmt::Display;
use std::iter::FromIterator;

use proc_macro::{Delimiter, Group, Ident, Literal, Punct, Spacing, Span, TokenStream, TokenTree};

use crate::{Kind, Radix, Range, Value};

type TokenIter = std::iter::Peekable<proc_macro::token_stream::IntoIter>;

pub(crate) struct SyntaxError {
    message: String,
    span: Span,
}

impl SyntaxError {
    pub(crate) fn into_compile_error(self) -> TokenStream {
        // compile_error! { $message }
        TokenStream::from_iter(vec![
            TokenTree::Ident(Ident::new("compile_error", self.span)),
            TokenTree::Punct({
                let mut punct = Punct::new('!', Spacing::Alone);
                punct.set_span(self.span);
                punct
            }),
            TokenTree::Group({
                let mut group = Group::new(Delimiter::Brace, {
                    TokenStream::from_iter(vec![TokenTree::Literal({
                        let mut string = Literal::string(&self.message);
                        string.set_span(self.span);
                        string
                    })])
                });
                group.set_span(self.span);
                group
            }),
        ])
    }
}

fn next_token(iter: &mut TokenIter) -> Result<TokenTree, SyntaxError> {
    iter.next().ok_or_else(|| SyntaxError {
        message: "unexpected end of input".to_owned(),
        span: Span::call_site(),
    })
}

fn next_token_if<F>(iter: &mut TokenIter, f: F) -> Result<Option<TokenTree>, SyntaxError>
where
    F: FnOnce(&TokenTree) -> bool,
{
    if iter.peek().is_none() {
        return Err(SyntaxError {
            message: "unexpected end of input".to_owned(),
            span: Span::call_site(),
        });
    }

    Ok(iter.next_if(f))
}

fn syntax<T: Borrow<TokenTree>, M: Display>(token: T, message: M) -> SyntaxError {
    SyntaxError {
        message: message.to_string(),
        span: token.borrow().span(),
    }
}

pub(crate) fn require_ident(iter: &mut TokenIter) -> Result<Ident, SyntaxError> {
    match next_token(iter)? {
        TokenTree::Ident(ident) => Ok(ident),
        other => Err(syntax(other, "expected ident")),
    }
}

pub(crate) fn require_keyword(iter: &mut TokenIter, keyword: &str) -> Result<(), SyntaxError> {
    let token = next_token(iter)?;
    if let TokenTree::Ident(ident) = &token {
        if ident.to_string() == keyword {
            return Ok(());
        }
    }
    Err(syntax(token, format!("expected `{}`", keyword)))
}

pub(crate) fn require_value_or_value_expr(iter: &mut TokenIter) -> Result<Value, SyntaxError> {
    let mut token = next_token(iter)?;
    let mut op = None;
    let mut rhs = None;

    loop {
        match token {
            TokenTree::Group(group) => {
                let delimiter = group.delimiter();
                let mut stream = group.stream().into_iter().peekable();
                token = TokenTree::Group(group);
                match delimiter {
                    Delimiter::None => {
                        let first = match stream.next() {
                            Some(first) => first,
                            None => break,
                        };
                        match stream.next() {
                            Some(_) => break,
                            None => token = first,
                        }
                    }
                    Delimiter::Parenthesis => {
                        // Check for `(value + literal)` or `(value - literal)`
                        let first = match stream.next() {
                            Some(first) => first,
                            None => break,
                        };
                        match next_token_if(&mut stream, |tt| matches!(tt, TokenTree::Punct(_)))? {
                            Some(TokenTree::Punct(p)) => match p.as_char() {
                                operator @ ('+' | '-') => op = Some(operator),
                                _ => {
                                    return Err(syntax(
                                        TokenTree::Punct(p),
                                        "unexpected operator, only '+' or '-' are supported",
                                    ));
                                }
                            },
                            _ => break,
                        }
                        let num = require_value(&mut stream)?;
                        rhs = Some(num);
                        match stream.next() {
                            Some(_) => break,
                            None => token = first,
                        }
                    }
                    _ => break,
                }
            }
            TokenTree::Literal(lit) => {
                let mut lhs = parse_literal(&lit).ok_or_else(|| {
                    let token = TokenTree::Literal(lit);
                    syntax(token, "expected unsuffixed integer literal")
                })?;
                return match (op, rhs) {
                    (None, _) => Ok(lhs),
                    (Some('+'), Some(rhs)) => {
                        assert_eq!(lhs.kind, rhs.kind);
                        lhs.int += rhs.int;
                        Ok(lhs)
                    }
                    (Some('-'), Some(rhs)) => {
                        assert_eq!(lhs.kind, rhs.kind);
                        lhs.int -= rhs.int;
                        Ok(lhs)
                    }
                    _ => unreachable!(),
                };
            }
            _ => break,
        }
    }

    Err(syntax(
        token,
        "expected integer or integer arithmetic expression",
    ))
}

pub(crate) fn require_value(iter: &mut TokenIter) -> Result<Value, SyntaxError> {
    let mut token = next_token(iter)?;

    loop {
        match token {
            TokenTree::Group(group) => {
                let delimiter = group.delimiter();
                let mut stream = group.stream().into_iter();
                token = TokenTree::Group(group);
                if delimiter != Delimiter::None {
                    break;
                }
                let first = match stream.next() {
                    Some(first) => first,
                    None => break,
                };
                match stream.next() {
                    Some(_) => break,
                    None => token = first,
                }
            }
            TokenTree::Literal(lit) => {
                return parse_literal(&lit).ok_or_else(|| {
                    let token = TokenTree::Literal(lit);
                    syntax(token, "expected unsuffixed integer literal")
                });
            }
            _ => break,
        }
    }

    Err(syntax(token, "expected integer"))
}

pub(crate) fn require_if_punct(iter: &mut TokenIter, ch: char) -> Result<bool, SyntaxError> {
    let present = match iter.clone().next() {
        Some(TokenTree::Punct(_)) => {
            require_punct(iter, ch)?;
            true
        }
        _ => false,
    };
    Ok(present)
}

pub(crate) fn require_punct(iter: &mut TokenIter, ch: char) -> Result<(), SyntaxError> {
    let token = next_token(iter)?;
    if let TokenTree::Punct(punct) = &token {
        if punct.as_char() == ch {
            return Ok(());
        }
    }
    Err(syntax(token, format!("expected `{}`", ch)))
}

pub(crate) fn require_braces(iter: &mut TokenIter) -> Result<TokenStream, SyntaxError> {
    let token = next_token(iter)?;
    if let TokenTree::Group(group) = &token {
        if group.delimiter() == Delimiter::Brace {
            return Ok(group.stream());
        }
    }
    Err(syntax(token, "expected curly braces"))
}

pub(crate) fn require_end(iter: &mut TokenIter) -> Result<(), SyntaxError> {
    match iter.next() {
        Some(token) => Err(syntax(token, "unexpected token")),
        None => Ok(()),
    }
}

pub(crate) fn validate_range(
    begin: Value,
    end: Value,
    inclusive: bool,
) -> Result<Range, SyntaxError> {
    let kind = if begin.kind == end.kind {
        begin.kind
    } else {
        let expected = match begin.kind {
            Kind::Int => "integer",
            Kind::Byte => "byte",
            Kind::Char => "character",
        };
        return Err(SyntaxError {
            message: format!("expected {} literal", expected),
            span: end.span,
        });
    };

    let suffix = if begin.suffix.is_empty() {
        end.suffix
    } else if end.suffix.is_empty() || begin.suffix == end.suffix {
        begin.suffix
    } else {
        return Err(SyntaxError {
            message: format!("expected suffix `{}`", begin.suffix),
            span: end.span,
        });
    };

    let radix = if begin.radix == end.radix {
        begin.radix
    } else if begin.radix == Radix::LowerHex && end.radix == Radix::UpperHex
        || begin.radix == Radix::UpperHex && end.radix == Radix::LowerHex
    {
        Radix::UpperHex
    } else {
        let expected = match begin.radix {
            Radix::Binary => "binary",
            Radix::Octal => "octal",
            Radix::Decimal => "base 10",
            Radix::LowerHex | Radix::UpperHex => "hexadecimal",
        };
        return Err(SyntaxError {
            message: format!("expected {} literal", expected),
            span: end.span,
        });
    };

    Ok(Range {
        begin: begin.int,
        end: end.int,
        inclusive,
        kind,
        suffix,
        width: cmp::min(begin.width, end.width),
        radix,
    })
}

fn parse_literal(lit: &Literal) -> Option<Value> {
    let span = lit.span();
    let repr = lit.to_string();
    assert!(!repr.starts_with('_'));

    if repr.starts_with("b'") && repr.ends_with('\'') && repr.len() == 4 {
        return Some(Value {
            int: repr.as_bytes()[2] as u64,
            kind: Kind::Byte,
            suffix: String::new(),
            width: 0,
            radix: Radix::Decimal,
            span,
        });
    }

    if repr.starts_with('\'') && repr.ends_with('\'') && repr.chars().count() == 3 {
        return Some(Value {
            int: repr[1..].chars().next().unwrap() as u64,
            kind: Kind::Char,
            suffix: String::new(),
            width: 0,
            radix: Radix::Decimal,
            span,
        });
    }

    let (mut radix, radix_n) = if repr.starts_with("0b") {
        (Radix::Binary, 2)
    } else if repr.starts_with("0o") {
        (Radix::Octal, 8)
    } else if repr.starts_with("0x") {
        (Radix::LowerHex, 16)
    } else if repr.starts_with("0X") {
        (Radix::UpperHex, 16)
    } else {
        (Radix::Decimal, 10)
    };

    let mut iter = repr.char_indices();
    let mut digits = String::new();
    let mut suffix = String::new();

    if radix != Radix::Decimal {
        let _ = iter.nth(1);
    }

    for (i, ch) in iter {
        match ch {
            '_' => continue,
            '0'..='9' => digits.push(ch),
            'A'..='F' if radix == Radix::LowerHex => {
                digits.push(ch);
                radix = Radix::UpperHex;
            }
            'a'..='f' | 'A'..='F' if radix_n == 16 => digits.push(ch),
            '.' => return None,
            _ => {
                if digits.is_empty() {
                    return None;
                }
                suffix = repr;
                suffix.replace_range(..i, "");
                break;
            }
        }
    }

    let int = u64::from_str_radix(&digits, radix_n).ok()?;
    let kind = Kind::Int;
    let width = digits.len();
    Some(Value {
        int,
        kind,
        suffix,
        width,
        radix,
        span,
    })
}

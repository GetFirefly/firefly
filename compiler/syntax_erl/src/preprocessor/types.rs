use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;

use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Symbol;

use crate::lexer::{AtomToken, IdentToken, SymbolToken};
use crate::lexer::{LexicalToken, Token};

use super::token_reader::{ReadFrom, TokenReader};
use super::{PreprocessorError, Result};

/// The list of tokens that can be used as a macro name.
#[derive(Debug, Clone)]
pub enum MacroName {
    Atom(AtomToken),
    Variable(IdentToken),
}
impl MacroName {
    /// Returns the value of this token.
    pub fn value(&self) -> Token {
        match *self {
            MacroName::Atom(ref token) => token.token(),
            MacroName::Variable(ref token) => token.token(),
        }
    }

    /// Returns the original textual representation of this token.
    pub fn symbol(&self) -> Symbol {
        match *self {
            MacroName::Atom(ref token) => token.symbol(),
            MacroName::Variable(ref token) => token.symbol(),
        }
    }

    pub fn span(&self) -> SourceSpan {
        match *self {
            MacroName::Atom(ref token) => token.span(),
            MacroName::Variable(ref token) => token.span(),
        }
    }
}
impl Eq for MacroName {}
impl PartialEq for MacroName {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}
impl Hash for MacroName {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.value().hash(hasher);
    }
}
impl fmt::Display for MacroName {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.symbol())
    }
}
impl ReadFrom for MacroName {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        if let Some(token) = reader.try_read()? {
            Ok(MacroName::Atom(token))
        } else {
            let token = reader.read()?;
            Ok(MacroName::Variable(token))
        }
    }
}

#[derive(Debug, Clone)]
pub struct MacroVariables {
    pub _open_paren: SymbolToken,
    pub list: List<IdentToken>,
    pub _close_paren: SymbolToken,
}
impl MacroVariables {
    /// Returns an iterator which iterates over this variables.
    pub fn iter(&self) -> ListIter<IdentToken> {
        self.list.iter()
    }

    /// Returns the number of this variables.
    pub fn len(&self) -> usize {
        self.list.iter().count()
    }

    /// Returns `true` if there are no variables.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self._open_paren.0, self._close_paren.2)
    }
}
impl Eq for MacroVariables {}
impl PartialEq for MacroVariables {
    fn eq(&self, other: &Self) -> bool {
        self.list.eq(&other.list)
    }
}
impl fmt::Display for MacroVariables {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})", self.list)
    }
}
impl ReadFrom for MacroVariables {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(MacroVariables {
            _open_paren: reader.read_expected(&Token::LParen)?,
            list: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
        })
    }
}

/// Macro arguments.
#[derive(Debug, Clone)]
pub struct MacroArgs {
    pub _open_paren: SymbolToken,
    pub list: List<MacroArg>,
    pub _close_paren: SymbolToken,
}
impl MacroArgs {
    /// Returns an iterator which iterates over this arguments.
    pub fn iter(&self) -> ListIter<MacroArg> {
        self.list.iter()
    }

    /// Returns the number of this arguments.
    pub fn len(&self) -> usize {
        self.list.iter().count()
    }

    /// Returns `true` if there are no arguments.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self._open_paren.0, self._close_paren.2)
    }
}
impl Eq for MacroArgs {}
impl PartialEq for MacroArgs {
    fn eq(&self, other: &Self) -> bool {
        self.list.eq(&other.list)
    }
}
impl fmt::Display for MacroArgs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "({})", self.list)
    }
}
impl ReadFrom for MacroArgs {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(MacroArgs {
            _open_paren: reader.read_expected(&Token::LParen)?,
            list: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
        })
    }
}

/// Macro argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroArg {
    /// Tokens which represent a macro argument.
    ///
    /// Note that this must not be empty.
    pub tokens: Vec<LexicalToken>,
}
impl MacroArg {
    pub fn span(&self) -> SourceSpan {
        let start = self.tokens.first().unwrap().span().start();
        let end = self.tokens.last().unwrap().span().end();
        SourceSpan::new(start, end)
    }
}
impl fmt::Display for MacroArg {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for t in &self.tokens {
            write!(f, "{}", t)?;
        }
        Ok(())
    }
}
impl ReadFrom for MacroArg {
    fn try_read_from<R, S>(reader: &mut R) -> Result<Option<Self>>
    where
        R: TokenReader<Source = S>,
    {
        let mut stack = Vec::new();
        let mut arg = Vec::new();
        while let Some(ref token @ LexicalToken(_, _, _)) = reader.try_read_token()? {
            match token.1 {
                Token::RParen if stack.is_empty() => {
                    reader.unread_token(token.clone().into());
                    return if arg.is_empty() {
                        Ok(None)
                    } else {
                        Ok(Some(MacroArg { tokens: arg }))
                    };
                }
                Token::RBrace | Token::RBracket | Token::BinaryEnd if stack.is_empty() => {
                    return Err(PreprocessorError::UnexpectedToken {
                        token: token.clone(),
                        expected: vec![Token::RParen.to_string()],
                    });
                }
                Token::Comma if stack.is_empty() => {
                    if arg.len() == 0 {
                        return Err(PreprocessorError::UnexpectedToken {
                            token: token.clone(),
                            expected: vec![],
                        });
                    }
                    reader.unread_token(token.clone().into());
                    return Ok(Some(MacroArg { tokens: arg }));
                }
                Token::LParen | Token::LBrace | Token::LBracket | Token::BinaryStart => {
                    stack.push(token.clone());
                }
                Token::RParen | Token::RBrace | Token::RBracket | Token::BinaryEnd => {
                    match stack.pop() {
                        None => unreachable!(),
                        Some(LexicalToken(_, t2, _)) => {
                            let closing = t2.get_closing_token();
                            if token.1 != closing {
                                return Err(PreprocessorError::UnexpectedToken {
                                    token: token.clone(),
                                    expected: vec![closing.to_string()],
                                });
                            }
                        }
                    }
                }
                _ => (),
            }
            arg.push(token.clone());
        }
        Err(PreprocessorError::UnexpectedEOF)
    }
}

/// Tail part of a linked list (cons cell).
#[derive(Debug, Clone)]
pub enum Tail<T> {
    Nil,
    Cons {
        _comma: SymbolToken,
        head: T,
        tail: Box<Tail<T>>,
    },
}
impl<T: PartialEq> Eq for Tail<T> {}
impl<T: PartialEq> PartialEq for Tail<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (
                Self::Cons {
                    head: lh, tail: lt, ..
                },
                Self::Cons {
                    head: rh, tail: rt, ..
                },
            ) => lh == rh && lt == rt,
            _ => false,
        }
    }
}
impl<T: fmt::Display> fmt::Display for Tail<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Tail::Nil => Ok(()),
            Tail::Cons {
                ref head, ref tail, ..
            } => write!(f, ",{}{}", head, tail),
        }
    }
}
impl<U: ReadFrom> ReadFrom for Tail<U> {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        if let Some(_comma) = reader.try_read_expected(&Token::Comma)? {
            let head = reader.read()?;
            let tail = Box::new(reader.read()?);
            Ok(Tail::Cons { _comma, head, tail })
        } else {
            Ok(Tail::Nil)
        }
    }
}

/// Linked list (cons cell).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum List<T> {
    Nil,
    Cons { head: T, tail: Tail<T> },
}
impl<T> List<T> {
    /// Returns an iterator which iterates over the elements in this list.
    pub fn iter(&self) -> ListIter<T> {
        ListIter(ListIterInner::List(self))
    }
}
impl<T: PartialEq> Eq for List<T> {}
impl<T: PartialEq> PartialEq for List<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Nil, Self::Nil) => true,
            (Self::Cons { head: lh, tail: lt }, Self::Cons { head: rh, tail: rt }) => {
                lh == rh && lt == rt
            }
            _ => false,
        }
    }
}
impl<T: fmt::Display> fmt::Display for List<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            List::Nil => Ok(()),
            List::Cons { ref head, ref tail } => write!(f, "{}{}", head, tail),
        }
    }
}
impl<U: ReadFrom> ReadFrom for List<U> {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        if let Some(head) = reader.try_read()? {
            let tail = reader.read()?;
            Ok(List::Cons { head, tail })
        } else {
            Ok(List::Nil)
        }
    }
}

/// An iterator which iterates over the elements in a `List`.
#[derive(Debug)]
pub struct ListIter<'a, T: 'a>(ListIterInner<'a, T>);
impl<'a, T: 'a> Iterator for ListIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

#[derive(Debug)]
enum ListIterInner<'a, T: 'a> {
    List(&'a List<T>),
    Tail(&'a Tail<T>),
    End,
}
impl<'a, T: 'a> Iterator for ListIterInner<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        match mem::replace(self, ListIterInner::End) {
            ListIterInner::List(&List::Cons { ref head, ref tail }) => {
                *self = ListIterInner::Tail(tail);
                Some(head)
            }
            ListIterInner::Tail(&Tail::Cons {
                ref head, ref tail, ..
            }) => {
                *self = ListIterInner::Tail(tail);
                Some(head)
            }
            ListIterInner::List(&List::Nil)
            | ListIterInner::Tail(&Tail::Nil)
            | ListIterInner::End => None,
        }
    }
}

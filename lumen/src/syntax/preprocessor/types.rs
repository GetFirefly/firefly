//! Miscellaneous types.
use std::hash::{Hash, Hasher};

use trackable::{
    track, track_assert, track_assert_eq, track_assert_ne, track_assert_some, track_panic,
};

use crate::syntax::tokenizer::tokens::{AtomToken, SymbolToken, VariableToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use super::token_reader::{ReadFrom, TokenReader};
use super::{ErrorKind, Result};

/// The list of tokens that can be used as a macro name.
#[derive(Debug, Clone)]
pub enum MacroName {
    Atom(AtomToken),
    Variable(VariableToken),
}
impl MacroName {
    /// Returns the value of this token.
    pub fn value(&self) -> &str {
        match *self {
            MacroName::Atom(ref t) => t.value(),
            MacroName::Variable(ref t) => t.value(),
        }
    }

    /// Returns the original textual representation of this token.
    pub fn text(&self) -> &str {
        match *self {
            MacroName::Atom(ref t) => t.text(),
            MacroName::Variable(ref t) => t.text(),
        }
    }
}
impl PartialEq for MacroName {
    fn eq(&self, other: &Self) -> bool {
        self.value() == other.value()
    }
}
impl Eq for MacroName {}
impl Hash for MacroName {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        self.value().hash(hasher);
    }
}
impl PositionRange for MacroName {
    fn start_position(&self) -> Position {
        match *self {
            MacroName::Atom(ref t) => t.start_position(),
            MacroName::Variable(ref t) => t.start_position(),
        }
    }
    fn end_position(&self) -> Position {
        match *self {
            MacroName::Atom(ref t) => t.end_position(),
            MacroName::Variable(ref t) => t.end_position(),
        }
    }
}
impl std::fmt::Display for MacroName {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.text())
    }
}
impl ReadFrom for MacroName {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        if let Some(token) = track!(reader.try_read())? {
            Ok(MacroName::Atom(token))
        } else {
            let token = track!(reader.read())?;
            Ok(MacroName::Variable(token))
        }
    }
}

/// Macro variables.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct MacroVariables {
    pub _open_paren: SymbolToken,
    pub list: List<VariableToken>,
    pub _close_paren: SymbolToken,
}
impl MacroVariables {
    /// Returns an iterator which iterates over this variables.
    pub fn iter(&self) -> ListIter<VariableToken> {
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
}
impl PositionRange for MacroVariables {
    fn start_position(&self) -> Position {
        self._open_paren.start_position()
    }
    fn end_position(&self) -> Position {
        self._close_paren.end_position()
    }
}
impl std::fmt::Display for MacroVariables {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({})", self.list)
    }
}
impl ReadFrom for MacroVariables {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(MacroVariables {
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            list: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
        })
    }
}

/// Macro arguments.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
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
}
impl PositionRange for MacroArgs {
    fn start_position(&self) -> Position {
        self._open_paren.start_position()
    }
    fn end_position(&self) -> Position {
        self._close_paren.end_position()
    }
}
impl std::fmt::Display for MacroArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "({})", self.list)
    }
}
impl ReadFrom for MacroArgs {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(MacroArgs {
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            list: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
        })
    }
}

/// Macro argument.
#[derive(Debug, Clone)]
pub struct MacroArg {
    /// Tokens which represent a macro argument.
    ///
    /// Note that this must not be empty.
    pub tokens: Vec<LexicalToken>,
}
impl PositionRange for MacroArg {
    fn start_position(&self) -> Position {
        self.tokens.first().as_ref().unwrap().start_position()
    }
    fn end_position(&self) -> Position {
        self.tokens.last().as_ref().unwrap().end_position()
    }
}
impl std::fmt::Display for MacroArg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for t in &self.tokens {
            write!(f, "{}", t.text())?;
        }
        Ok(())
    }
}
impl ReadFrom for MacroArg {
    fn try_read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Option<Self>>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        let mut stack = Vec::new();
        let mut arg = Vec::new();
        while let Some(token) = track!(reader.try_read_token())? {
            if let LexicalToken::Symbol(ref s) = token {
                match s.value() {
                    Symbol::CloseParen if stack.is_empty() => {
                        reader.unread_token(s.clone().into());
                        return if arg.is_empty() {
                            Ok(None)
                        } else {
                            Ok(Some(MacroArg { tokens: arg }))
                        };
                    }
                    Symbol::Comma if stack.is_empty() => {
                        track_assert_ne!(arg.len(), 0, ErrorKind::InvalidInput);
                        reader.unread_token(s.clone().into());
                        return Ok(Some(MacroArg { tokens: arg }));
                    }
                    Symbol::OpenParen
                    | Symbol::OpenBrace
                    | Symbol::OpenSquare
                    | Symbol::DoubleLeftAngle => {
                        stack.push(s.clone());
                    }
                    Symbol::CloseParen
                    | Symbol::CloseBrace
                    | Symbol::CloseSquare
                    | Symbol::DoubleRightAngle => {
                        let last = track_assert_some!(stack.pop(), ErrorKind::InvalidInput);
                        let expected = match last.value() {
                            Symbol::OpenParen => Symbol::CloseParen,
                            Symbol::OpenBrace => Symbol::CloseBrace,
                            Symbol::OpenSquare => Symbol::CloseSquare,
                            Symbol::DoubleLeftAngle => Symbol::DoubleRightAngle,
                            _ => unreachable!(),
                        };
                        track_assert_eq!(s.value(), expected, ErrorKind::InvalidInput);
                    }
                    _ => {}
                }
            }
            arg.push(token);
        }
        track_panic!(ErrorKind::UnexpectedEos);
    }
}

/// Tail part of a linked list (cons cell).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum Tail<T> {
    Null,
    Cons {
        _comma: SymbolToken,
        head: T,
        tail: Box<Tail<T>>,
    },
}
impl<T: std::fmt::Display> std::fmt::Display for Tail<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Tail::Null => Ok(()),
            Tail::Cons {
                ref head, ref tail, ..
            } => write!(f, ",{}{}", head, tail),
        }
    }
}
impl<U: ReadFrom> ReadFrom for Tail<U> {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        if let Some(_comma) = track!(reader.try_read_expected(&Symbol::Comma))? {
            let head = track!(reader.read())?;
            let tail = Box::new(track!(reader.read())?);
            Ok(Tail::Cons { _comma, head, tail })
        } else {
            Ok(Tail::Null)
        }
    }
}

/// Linked list (cons cell).
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum List<T> {
    Null,
    Cons { head: T, tail: Tail<T> },
}
impl<T> List<T> {
    /// Returns an iterator which iterates over the elements in this list.
    pub fn iter(&self) -> ListIter<T> {
        ListIter(ListIterInner::List(self))
    }
}
impl<T: std::fmt::Display> std::fmt::Display for List<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            List::Null => Ok(()),
            List::Cons { ref head, ref tail } => write!(f, "{}{}", head, tail),
        }
    }
}
impl<U: ReadFrom> ReadFrom for List<U> {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = ::std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        if let Some(head) = track!(reader.try_read())? {
            let tail = track!(reader.read())?;
            Ok(List::Cons { head, tail })
        } else {
            Ok(List::Null)
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
        match std::mem::replace(self, ListIterInner::End) {
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
            ListIterInner::List(&List::Null)
            | ListIterInner::Tail(&Tail::Null)
            | ListIterInner::End => None,
        }
    }
}

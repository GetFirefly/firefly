use std::collections::HashMap;
use std::fmt;

use firefly_diagnostics::SourceSpan;
use firefly_intern::Symbol;

use crate::lexer::{DelayedSubstitution, LexicalToken, Token};
use crate::lexer::{IdentToken, SymbolToken};

use super::directives::Define;
use super::token_reader::{ReadFrom, TokenReader};
use super::types::{MacroArgs, MacroName};
use super::Result;

pub enum MacroIdent {
    Const(Symbol),
    Func(Symbol, usize),
}
impl MacroIdent {
    pub fn ident(&self) -> Symbol {
        match self {
            MacroIdent::Const(sym) => *sym,
            MacroIdent::Func(sym, _) => *sym,
        }
    }

    pub fn arity(&self) -> Option<usize> {
        match self {
            MacroIdent::Const(_) => None,
            MacroIdent::Func(_, arity) => Some(*arity),
        }
    }
}

impl From<&MacroCall> for MacroIdent {
    fn from(call: &MacroCall) -> MacroIdent {
        let ident = match &call.name {
            MacroName::Atom(tok) => tok.symbol(),
            MacroName::Variable(tok) => tok.symbol(),
        };

        if let Some(args) = &call.args {
            MacroIdent::Func(ident, args.len())
        } else {
            MacroIdent::Const(ident)
        }
    }
}

impl From<&super::directives::Define> for MacroIdent {
    fn from(def: &super::directives::Define) -> MacroIdent {
        let ident = match &def.name {
            MacroName::Atom(tok) => tok.symbol(),
            MacroName::Variable(tok) => tok.symbol(),
        };

        if let Some(args) = &def.variables {
            MacroIdent::Func(ident, args.len())
        } else {
            MacroIdent::Const(ident)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MacroContainer {
    func_defines: HashMap<Symbol, HashMap<usize, MacroDef>>,
    const_defines: HashMap<Symbol, MacroDef>,
}
impl MacroContainer {
    pub fn new() -> Self {
        MacroContainer {
            func_defines: HashMap::new(),
            const_defines: HashMap::new(),
        }
    }

    pub fn insert<T>(&mut self, key: T, def: MacroDef) -> bool
    where
        T: Into<MacroIdent>,
    {
        let key: MacroIdent = key.into();
        match key {
            MacroIdent::Const(name) => self.const_defines.insert(name, def).is_some(),
            MacroIdent::Func(name, arity) => {
                if !self.func_defines.contains_key(&name) {
                    self.func_defines.insert(name, HashMap::new());
                }
                let container = self.func_defines.get_mut(&name).unwrap();
                container.insert(arity, def).is_some()
            }
        }
    }

    pub fn get<'a, T>(&'a self, key: T) -> Option<&'a MacroDef>
    where
        T: Into<MacroIdent>,
    {
        let key: MacroIdent = key.into();
        match key {
            MacroIdent::Const(name) => self.const_defines.get(&name),
            MacroIdent::Func(name, arity) => {
                self.func_defines.get(&name).and_then(|c| c.get(&arity))
            }
        }
    }

    pub fn undef(&mut self, symbol: &Symbol) -> bool {
        let mut res = false;
        res |= self.const_defines.remove(symbol).is_some();
        res |= self.func_defines.remove(symbol).is_some();
        res
    }

    pub fn defined(&self, symbol: &Symbol) -> bool {
        self.defined_const(symbol) || self.defined_func(symbol)
    }
    pub fn defined_const(&self, symbol: &Symbol) -> bool {
        self.const_defines.contains_key(symbol)
    }
    pub fn defined_func(&self, symbol: &Symbol) -> bool {
        self.func_defines.contains_key(symbol)
    }
}

/// Macro Definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MacroDef {
    Boolean(bool),
    Atom(Symbol),
    String(Symbol),
    Static(Define),
    Dynamic(Vec<LexicalToken>),
    DelayedSubstitution(DelayedSubstitution),
}
impl MacroDef {
    /// Returns `true` if this macro has variables, otherwise `false`.
    pub fn has_variables(&self) -> bool {
        match *self {
            MacroDef::Static(ref d) => d.variables.is_some(),
            MacroDef::Dynamic(_) => false,
            MacroDef::Atom(_) => false,
            MacroDef::String(_) => false,
            MacroDef::Boolean(_) => false,
            MacroDef::DelayedSubstitution(_) => false,
        }
    }
}

/// Macro call.
#[derive(Debug, Clone)]
pub struct MacroCall {
    pub _question: SymbolToken,
    pub name: MacroName,
    pub args: Option<MacroArgs>,
}
impl MacroCall {
    pub fn span(&self) -> SourceSpan {
        let start = self._question.0;
        let end = self
            .args
            .as_ref()
            .map(|a| a.span().end())
            .unwrap_or_else(|| self.name.span().end());
        SourceSpan::new(start, end)
    }

    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl fmt::Display for MacroCall {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "?{}{}",
            self.name.symbol(),
            self.args.as_ref().map_or("".to_string(), |a| a.to_string())
        )
    }
}
impl ReadFrom for MacroCall {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(MacroCall {
            _question: reader.read_expected(&Token::Question)?,
            name: reader.read()?,
            args: reader.try_read()?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct NoArgsMacroCall {
    pub _question: SymbolToken,
    pub name: MacroName,
}
impl NoArgsMacroCall {
    pub fn span(&self) -> SourceSpan {
        SourceSpan::new(self._question.span().start(), self.name.span().end())
    }
}
impl ReadFrom for NoArgsMacroCall {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(NoArgsMacroCall {
            _question: reader.read_expected(&Token::Question)?,
            name: reader.read()?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Stringify {
    pub _double_question: SymbolToken,
    pub name: IdentToken,
}
impl Stringify {
    pub fn span(&self) -> SourceSpan {
        let start = self._double_question.0;
        let end = self.name.2;
        SourceSpan::new(start, end)
    }
}
impl fmt::Display for Stringify {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "??{}", self.name)
    }
}
impl ReadFrom for Stringify {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Stringify {
            _double_question: reader.read_expected(&Token::DoubleQuestion)?,
            name: reader.read()?,
        })
    }
}

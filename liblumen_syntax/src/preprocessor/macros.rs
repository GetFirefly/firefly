use std::fmt;

use liblumen_diagnostics::ByteSpan;

use crate::lexer::{LexicalToken, Token, Symbol};
use crate::lexer::{SymbolToken, IdentToken};

use super::Result;
use super::directives::Define;
use super::token_reader::{TokenReader, ReadFrom};
use super::types::{MacroName, MacroArgs};

/// Macro Definition.
#[derive(Debug, Clone)]
pub enum MacroDef {
    Boolean(bool),
    String(Symbol),
    Static(Define),
    Dynamic(Vec<LexicalToken>),
}
impl MacroDef {
    /// Returns `true` if this macro has variables, otherwise `false`.
    pub fn has_variables(&self) -> bool {
        match *self {
            MacroDef::Static(ref d) => d.variables.is_some(),
            MacroDef::Dynamic(_) => false,
            MacroDef::String(_) => false,
            MacroDef::Boolean(_) => false,
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
    pub fn span(&self) -> ByteSpan {
        let start = self._question.0;
        let end = self.args
            .as_ref()
            .map(|a| a.span().end())
            .unwrap_or_else(|| self.name.span().end());
        ByteSpan::new(start, end)
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
    pub fn span(&self) -> ByteSpan {
        ByteSpan::new(self._question.span().start(), self.name.span().end())
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
    pub fn span(&self) -> ByteSpan {
        let start = self._double_question.0;
        let end = self.name.2;
        ByteSpan::new(start, end)
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

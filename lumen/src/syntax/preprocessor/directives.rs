//! Macro directives.
use std::collections::VecDeque;
use std::path::{Component, PathBuf};

use glob::glob;

use trackable::{track, track_assert, track_assert_some, track_panic};

use crate::syntax;
use crate::syntax::tokenizer::tokens::{AtomToken, StringToken, SymbolToken};
use crate::syntax::tokenizer::values::Symbol;
use crate::syntax::tokenizer::{LexicalToken, Position, PositionRange};

use super::token_reader::{ReadFrom, TokenReader};
use super::types::{MacroName, MacroVariables};
use super::util;
use super::{ErrorKind, Result};

/// `include` directive.
///
/// See [9.1 File Inclusion](http://erlang.org/doc/reference_manual/macros.html#id85412)
/// for detailed information.
#[derive(Debug, Clone)]
pub struct Include {
    pub _hyphen: SymbolToken,
    pub _include: AtomToken,
    pub _open_paren: SymbolToken,
    pub path: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Include {
    /// Executes file inclusion.
    pub fn include(&self) -> Result<(PathBuf, String)> {
        let path = track!(util::substitute_path_variables(self.path.value()))?;
        let text = track!(util::read_file(&path))?;
        Ok((path, text))
    }
}
impl PositionRange for Include {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Include {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-include({}).", self.path.text())
    }
}
impl ReadFrom for Include {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<syntax::preprocessor::Error>,
    {
        Ok(Include {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _include: track!(reader.read_expected("include"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            path: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `include_lib` directive.
///
/// See [9.1 File Inclusion](http://erlang.org/doc/reference_manual/macros.html#id85412)
/// for detailed information.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct IncludeLib {
    pub _hyphen: SymbolToken,
    pub _include_lib: AtomToken,
    pub _open_paren: SymbolToken,
    pub path: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl IncludeLib {
    /// Executes file inclusion.
    pub fn include_lib(&self, code_paths: &VecDeque<PathBuf>) -> Result<(PathBuf, String)> {
        let mut path = track!(util::substitute_path_variables(self.path.value()))?;

        let temp_path = path.clone();
        let mut components = temp_path.components();
        if let Some(Component::Normal(app_name)) = components.next() {
            let app_name = track_assert_some!(app_name.to_str(), ErrorKind::InvalidInput);
            let pattern = format!("{}-*", app_name);
            'root: for root in code_paths.iter() {
                let pattern = root.join(&pattern);
                let pattern = track_assert_some!(pattern.to_str(), ErrorKind::InvalidInput);
                if let Some(entry) = track!(glob(pattern).map_err(super::Error::from))?.nth(0) {
                    path = track!(entry.map_err(super::Error::from))?;
                    for c in components {
                        path.push(c.as_os_str());
                    }
                    break 'root;
                }
            }
        }

        let text = track!(util::read_file(&path))?;
        Ok((path, text))
    }
}
impl PositionRange for IncludeLib {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for IncludeLib {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-include_lib({}).", self.path.text())
    }
}
impl ReadFrom for IncludeLib {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(IncludeLib {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _include_lib: track!(reader.read_expected("include_lib"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            path: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `error` directive.
///
/// See [9.6 -error() and -warning() directives][error_and_warning]
/// for detailed information.
///
/// [error_and_warning]: http://erlang.org/doc/reference_manual/macros.html#id85997
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Error {
    pub _hyphen: SymbolToken,
    pub _error: AtomToken,
    pub _open_paren: SymbolToken,
    pub message: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Error {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-error({}).", self.message.text())
    }
}
impl ReadFrom for Error {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Error {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _error: track!(reader.read_expected("error"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            message: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `warning` directive.
///
/// See [9.6 -error() and -warning() directives][error_and_warning]
/// for detailed information.
///
/// [error_and_warning]: http://erlang.org/doc/reference_manual/macros.html#id85997
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Warning {
    pub _hyphen: SymbolToken,
    pub _warning: AtomToken,
    pub _open_paren: SymbolToken,
    pub message: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Warning {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-warning({}).", self.message.text())
    }
}
impl ReadFrom for Warning {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Warning {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _warning: track!(reader.read_expected("warning"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            message: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `endif` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Endif {
    pub _hyphen: SymbolToken,
    pub _endif: AtomToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Endif {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Endif {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-endif.")
    }
}
impl ReadFrom for Endif {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Endif {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _endif: track!(reader.read_expected("endif"))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `else` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Else {
    pub _hyphen: SymbolToken,
    pub _else: AtomToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Else {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Else {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-else.")
    }
}
impl ReadFrom for Else {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Else {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _else: track!(reader.read_expected("else"))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `undef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Undef {
    pub _hyphen: SymbolToken,
    pub _undef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Undef {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Undef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-undef({}).", self.name.text())
    }
}
impl ReadFrom for Undef {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Undef {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _undef: track!(reader.read_expected("undef"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            name: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `ifdef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Ifdef {
    pub _hyphen: SymbolToken,
    pub _ifdef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Ifdef {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Ifdef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-ifdef({}).", self.name.text())
    }
}
impl ReadFrom for Ifdef {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Ifdef {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _ifdef: track!(reader.read_expected("ifdef"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            name: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `ifndef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Ifndef {
    pub _hyphen: SymbolToken,
    pub _ifndef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Ifndef {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Ifndef {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "-ifndef({}).", self.name.text())
    }
}
impl ReadFrom for Ifndef {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        Ok(Ifndef {
            _hyphen: track!(reader.read_expected(&Symbol::Hyphen))?,
            _ifndef: track!(reader.read_expected("ifndef"))?,
            _open_paren: track!(reader.read_expected(&Symbol::OpenParen))?,
            name: track!(reader.read())?,
            _close_paren: track!(reader.read_expected(&Symbol::CloseParen))?,
            _dot: track!(reader.read_expected(&Symbol::Dot))?,
        })
    }
}

/// `define` directive.
///
/// See [9.2 Defining and Using Macros][define_and_use] for detailed information.
///
/// [define_and_use]: http://erlang.org/doc/reference_manual/macros.html#id85572
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct Define {
    pub _hyphen: SymbolToken,
    pub _define: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub variables: Option<MacroVariables>,
    pub _comma: SymbolToken,
    pub replacement: Vec<LexicalToken>,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl PositionRange for Define {
    fn start_position(&self) -> Position {
        self._hyphen.start_position()
    }
    fn end_position(&self) -> Position {
        self._dot.end_position()
    }
}
impl std::fmt::Display for Define {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "-define({}{}, {}).",
            self.name,
            self.variables
                .as_ref()
                .map_or("".to_string(), |v| v.to_string(),),
            self.replacement
                .iter()
                .map(|t| t.text())
                .collect::<String>()
        )
    }
}
impl ReadFrom for Define {
    fn read_from<T, E>(reader: &mut TokenReader<T, E>) -> Result<Self>
    where
        T: Iterator<Item = std::result::Result<LexicalToken, E>>,
        E: Into<super::Error>,
    {
        let _hyphen = track!(reader.read_expected(&Symbol::Hyphen))?;
        let _define = track!(reader.read_expected("define"))?;
        let _open_paren = track!(reader.read_expected(&Symbol::OpenParen))?;
        let name = track!(reader.read())?;
        let variables = if let Some(token) =
            track!(reader.try_read_expected::<SymbolToken>(&Symbol::OpenParen,))?
        {
            reader.unread_token(token.into());
            Some(track!(reader.read())?)
        } else {
            None
        };
        let _comma = track!(reader.read_expected(&Symbol::Comma))?;

        let mut replacement = Vec::new();
        loop {
            if let Some(_close_paren) = track!(reader.try_read_expected(&Symbol::CloseParen))? {
                if let Some(_dot) = track!(reader.try_read_expected(&Symbol::Dot))? {
                    return Ok(Define {
                        _hyphen,
                        _define,
                        _open_paren,
                        name,
                        variables,
                        _comma,
                        replacement,
                        _close_paren,
                        _dot,
                    });
                }
                replacement.push(_close_paren.into());
            } else {
                let token = track!(reader.read_token())?;
                track_assert!(
                    token
                        .as_symbol_token()
                        .map_or(true, |s| s.value() != Symbol::Dot,),
                    ErrorKind::InvalidInput
                );
                replacement.push(token);
            }
        }
    }
}

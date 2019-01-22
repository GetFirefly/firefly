use std::collections::VecDeque;
use std::fmt;
use std::path::{Component, Path, PathBuf};

use glob::glob;
use liblumen_diagnostics::ByteSpan;

use crate::lexer::{symbols, Lexed, LexicalToken, Symbol, Token};
use crate::lexer::{AtomToken, StringToken, SymbolToken};
use crate::util;

use super::token_reader::{ReadFrom, TokenReader};
use super::types::{MacroName, MacroVariables};
use super::{PreprocessorError, Result};

/// `module` directive.
///
/// Not really a directive, but we need it for the ?MODULE macro
#[derive(Debug, Clone)]
pub struct Module {
    pub _hyphen: SymbolToken,
    pub _module: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: AtomToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Module {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }

    pub fn expand(&self) -> VecDeque<LexicalToken> {
        vec![
            self._hyphen.clone().into(),
            self._module.clone().into(),
            self._open_paren.clone().into(),
            self.name.clone().into(),
            self._close_paren.clone().into(),
            self._dot.clone().into(),
        ]
        .into()
    }
}
impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-module({}).", self.name.symbol())
    }
}
impl ReadFrom for Module {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Module {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _module: reader.read_expected(&symbols::Module)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            name: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

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
    pub fn include(&self) -> PathBuf {
        Path::new(&self.path.symbol().as_str().get()).to_path_buf()
    }

    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Include {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-include({}).", self.path.symbol())
    }
}
impl ReadFrom for Include {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Include {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _include: reader.read_expected(&symbols::Include)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            path: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
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
    pub fn include_lib(&self, code_paths: &VecDeque<PathBuf>) -> Result<PathBuf> {
        let mut path = match util::substitute_path_variables(self.path.symbol().as_str().get()) {
            Ok(path) => path,
            Err(err) => return Err(err.into()),
        };

        let temp_path = path.clone();
        let mut components = temp_path.components();
        if let Some(Component::Normal(app_name)) = components.next() {
            let app_name = app_name
                .to_str()
                .expect("internal error: expected app name here");
            let pattern = format!("{}-*", app_name);
            'root: for root in code_paths.iter() {
                let pattern = root.join(&pattern);
                let pattern = pattern.to_str().unwrap();
                if let Some(entry) = glob(pattern).map_err(PreprocessorError::from)?.nth(0) {
                    path = entry.map_err(PreprocessorError::from)?;
                    for c in components {
                        path.push(c.as_os_str());
                    }
                    break 'root;
                }
            }
        }
        Ok(path)
    }

    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for IncludeLib {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-include_lib({}).", self.path.symbol())
    }
}
impl ReadFrom for IncludeLib {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(IncludeLib {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _include_lib: reader.read_expected(&symbols::IncludeLib)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            path: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
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
pub struct Error {
    pub _hyphen: SymbolToken,
    pub _error: AtomToken,
    pub _open_paren: SymbolToken,
    pub message: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Error {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-error({}).", self.message.symbol())
    }
}
impl ReadFrom for Error {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Error {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _error: reader.read_expected(&symbols::Error)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            message: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
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
pub struct Warning {
    pub _hyphen: SymbolToken,
    pub _warning: AtomToken,
    pub _open_paren: SymbolToken,
    pub message: StringToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Warning {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Warning {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-warning({}).", self.message.symbol())
    }
}
impl ReadFrom for Warning {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Warning {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _warning: reader.read_expected(&symbols::Warning)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            message: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `endif` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Endif {
    pub _hyphen: SymbolToken,
    pub _endif: AtomToken,
    pub _dot: SymbolToken,
}
impl Endif {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Endif {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-endif.")
    }
}
impl ReadFrom for Endif {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Endif {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _endif: reader.read_expected(&symbols::Endif)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `else` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Else {
    pub _hyphen: SymbolToken,
    pub _else: AtomToken,
    pub _dot: SymbolToken,
}
impl Else {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Else {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-else.")
    }
}
impl ReadFrom for Else {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Else {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _else: reader.read_expected(&symbols::Else)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `undef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Undef {
    pub _hyphen: SymbolToken,
    pub _undef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Undef {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl fmt::Display for Undef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-undef({}).", self.name.symbol())
    }
}
impl ReadFrom for Undef {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Undef {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _undef: reader.read_expected(&symbols::Undef)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            name: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `if` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct If {
    pub _hyphen: SymbolToken,
    pub _if: AtomToken,
    pub _open_paren: SymbolToken,
    pub condition: VecDeque<Lexed>,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl If {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for If {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-if({:?}).", self.condition)
    }
}
impl ReadFrom for If {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(If {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _if: reader.read_expected(&symbols::If)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            condition: read_condition(reader)?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `elif` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Elif {
    pub _hyphen: SymbolToken,
    pub _elif: AtomToken,
    pub _open_paren: SymbolToken,
    pub condition: VecDeque<Lexed>,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Elif {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Elif {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-elif({:?}).", self.condition)
    }
}
impl ReadFrom for Elif {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Elif {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _elif: reader.read_expected(&symbols::Elif)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            condition: read_condition(reader)?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

fn read_condition<R, S>(reader: &mut R) -> Result<VecDeque<Lexed>>
where
    R: TokenReader<Source = S>,
{
    let mut open = 0;
    let mut condition = VecDeque::new();

    loop {
        match reader.try_read_token()? {
            None => return Err(PreprocessorError::UnexpectedEOF),
            Some(token) => match token {
                LexicalToken(_, Token::LParen, _) => {
                    open = open + 1;
                    condition.push_back(Ok(token));
                }
                LexicalToken(_, Token::RParen, _) if open == 0 => {
                    reader.unread_token(token);
                    break;
                }
                LexicalToken(_, Token::RParen, _) => {
                    open = open - 1;
                    condition.push_back(Ok(token));
                }
                _ => {
                    condition.push_back(Ok(token));
                }
            },
        }
    }

    Ok(condition)
}

/// `ifdef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Ifdef {
    pub _hyphen: SymbolToken,
    pub _ifdef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Ifdef {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl fmt::Display for Ifdef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-ifdef({}).", self.name.symbol())
    }
}
impl ReadFrom for Ifdef {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Ifdef {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _ifdef: reader.read_expected(&symbols::Ifdef)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            name: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `ifndef` directive.
///
/// See [9.5 Flow Control in Macros][flow_control] for detailed information.
///
/// [flow_control]: http://erlang.org/doc/reference_manual/macros.html#id85859
#[derive(Debug, Clone)]
pub struct Ifndef {
    pub _hyphen: SymbolToken,
    pub _ifndef: AtomToken,
    pub _open_paren: SymbolToken,
    pub name: MacroName,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl Ifndef {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        ByteSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl fmt::Display for Ifndef {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-ifndef({}).", self.name.symbol())
    }
}
impl ReadFrom for Ifndef {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(Ifndef {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _ifndef: reader.read_expected(&symbols::Ifndef)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            name: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

/// `define` directive.
///
/// See [9.2 Defining and Using Macros][define_and_use] for detailed information.
///
/// [define_and_use]: http://erlang.org/doc/reference_manual/macros.html#id85572
#[derive(Debug, Clone)]
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
impl Define {
    pub fn span(&self) -> ByteSpan {
        let start = self._hyphen.0;
        let end = self._dot.0;
        ByteSpan::new(start, end)
    }
}
impl fmt::Display for Define {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "-define({}{}, {}).",
            self.name,
            self.variables
                .as_ref()
                .map_or("".to_string(), |v| v.to_string(),),
            self.replacement
                .iter()
                .map(|t| t.to_string())
                .collect::<String>()
        )
    }
}
impl ReadFrom for Define {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        let _hyphen = reader.read_expected(&Token::Minus)?;
        let _define = reader.read_expected(&symbols::Define)?;
        let _open_paren = reader.read_expected(&Token::LParen)?;
        let name = reader.read()?;
        let variables =
            if let Some(token) = reader.try_read_expected::<SymbolToken>(&Token::LParen)? {
                reader.unread_token(token.into());
                Some(reader.read()?)
            } else {
                None
            };
        let _comma = reader.read_expected(&Token::Comma)?;

        let mut replacement = Vec::new();
        loop {
            if let Some(_close_paren) = reader.try_read_expected(&Token::RParen)? {
                if let Some(_dot) = reader.try_read_expected(&Token::Dot)? {
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
                match reader.read_token()? {
                    token @ LexicalToken(_, Token::Dot, _) => {
                        return Err(PreprocessorError::UnexpectedToken(token, Vec::new()));
                    }
                    token => replacement.push(token),
                }
            }
        }
    }
}

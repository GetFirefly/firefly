use std::collections::VecDeque;
use std::fmt;
use std::path::{Component, PathBuf};

use firefly_diagnostics::{Diagnostic, Label, SourceSpan, ToDiagnostic};
use firefly_intern::{symbols, Symbol};
use firefly_parser::substitute_path_variables;
use firefly_parser::PathVariableSubstituteError;

use crate::lexer::{AtomToken, IntegerToken, StringToken, SymbolToken};
use crate::lexer::{Lexed, LexicalToken, Token};

use super::token_reader::{ReadFrom, TokenReader};
use super::types::{MacroName, MacroVariables};
use super::{PreprocessorError, Result};

#[derive(Debug)]
pub enum IncludeLibErrorVariant {
    NoAppNameComponent,
    NotFound { searched: Vec<String> },
}

#[derive(Debug, thiserror::Error)]
pub enum DirectiveError {
    #[error("{source}")]
    PathSubstitute {
        span: SourceSpan,
        #[source]
        source: PathVariableSubstituteError,
    },

    #[error("could not find file")]
    FileNotFound {
        span: SourceSpan,
        searched: Vec<String>,
    },

    #[error("include_lib could not find file")]
    IncludeLibError {
        span: SourceSpan,
        first_searched: Vec<String>,
        second: IncludeLibErrorVariant,
    },
}
impl ToDiagnostic for DirectiveError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            DirectiveError::PathSubstitute { span, source } => Diagnostic::error()
                .with_message(source.to_string())
                .with_labels(vec![Label::primary(span.source_id(), span)
                    .with_message("in expansion of this path")]),
            DirectiveError::FileNotFound { span, searched } => {
                let aux_msg = if searched.is_empty() {
                    format!("attempted searching include paths, but none were specified\n")
                } else {
                    let mut msg = format!("attempted searching search paths:\n");
                    for path in searched.iter() {
                        msg.push_str(path);
                        msg.push('\n');
                    }
                    msg
                };

                Diagnostic::error()
                    .with_message("could not find file")
                    .with_labels(vec![
                        Label::primary(span.source_id(), span).with_message("failed to find file")
                    ])
                    .with_notes(vec![aux_msg])
            }
            DirectiveError::IncludeLibError {
                span,
                first_searched,
                second,
            } => {
                let aux_msg_1 = if first_searched.is_empty() {
                    format!("first, attempted searching include paths, but none were specified\n")
                } else {
                    let mut msg = format!("first, attempted searching include paths:\n");
                    for path in first_searched.iter() {
                        msg.push_str(" - ");
                        msg.push_str(path);
                        msg.push('\n');
                    }
                    msg
                };

                let aux_msg_2 = match second {
                    IncludeLibErrorVariant::NoAppNameComponent => {
                        format!("then, attempted searching codepath, but first path component wasn't an application directory!")
                    }
                    IncludeLibErrorVariant::NotFound { searched } if searched.is_empty() => {
                        format!("then, attempted searching codepath, but none were specified\n")
                    }
                    IncludeLibErrorVariant::NotFound { searched } => {
                        let mut msg = format!("then, attempted include from codepath:\n");

                        if searched.is_empty() {
                            msg.push_str("[]\n");
                        }
                        for path in searched.iter() {
                            msg.push_str(" - ");
                            msg.push_str(path);
                            msg.push('\n');
                        }

                        msg
                    }
                };

                Diagnostic::error()
                    .with_message("could not find file")
                    .with_labels(vec![
                        Label::primary(span.source_id(), span).with_message("failed to find file")
                    ])
                    .with_notes(vec![aux_msg_1, aux_msg_2])
            }
        }
    }
}

type DirectiveResult<T> = std::result::Result<T, DirectiveError>;

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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }

    pub fn expand(&self) -> VecDeque<LexicalToken> {
        let mod_span = self._module.span();
        vec![
            self._hyphen.clone().into(),
            LexicalToken(mod_span.start(), Token::Module, mod_span.end()),
            self._open_paren.clone().into(),
            self.name.clone().into(),
            self._close_paren.clone().into(),
            self._dot.clone().into(),
        ]
        .into()
    }
}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        self.name.symbol() == other.name.symbol()
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

fn do_include(
    subs_path: &PathBuf,
    include_paths: &VecDeque<PathBuf>,
) -> std::result::Result<PathBuf, Vec<String>> {
    let mut tmp_path = PathBuf::new();
    for include_path in include_paths.iter() {
        tmp_path.push(include_path);
        tmp_path.push(subs_path);
        if tmp_path.exists() {
            return Ok(tmp_path);
        }
        tmp_path.clear();
    }

    let searched: Vec<String> = include_paths
        .iter()
        .map(|path| {
            path.to_str()
                .map(|v| v.to_owned())
                .unwrap_or_else(|| path.to_string_lossy().chars().collect())
        })
        .collect();
    Err(searched)
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
    pub fn include(&self, include_paths: &VecDeque<PathBuf>) -> DirectiveResult<PathBuf> {
        let path =
            substitute_path_variables(self.path.symbol().as_str().get()).map_err(|source| {
                DirectiveError::PathSubstitute {
                    span: self.path.span(),
                    source,
                }
            })?;

        match do_include(&path, include_paths) {
            Ok(path) => Ok(path),
            Err(searched) => Err(DirectiveError::FileNotFound {
                span: self.span(),
                searched,
            }),
        }
    }

    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Include {}
impl PartialEq for Include {
    fn eq(&self, other: &Self) -> bool {
        self.path.symbol() == other.path.symbol()
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
    pub fn include_lib(
        &self,
        include_paths: &VecDeque<PathBuf>,
        code_paths: &VecDeque<PathBuf>,
    ) -> DirectiveResult<PathBuf> {
        let path =
            substitute_path_variables(self.path.symbol().as_str().get()).map_err(|source| {
                DirectiveError::PathSubstitute {
                    span: self.path.span(),
                    source,
                }
            })?;

        let first_searched = match do_include(&path, include_paths) {
            Ok(path) => return Ok(path),
            Err(searched) => searched,
        };

        let mut second_searched = Vec::new();

        let components: Vec<_> = path.components().collect();
        if let Component::Normal(_app_name) = &components[0] {
            for root in code_paths.iter() {
                let full_path = root.join(&path);
                if full_path.exists() {
                    return Ok(full_path);
                }

                let string = full_path
                    .to_str()
                    .map(|v| v.to_owned())
                    .unwrap_or_else(|| path.to_string_lossy().chars().collect());
                second_searched.push(string);
            }

            Err(DirectiveError::IncludeLibError {
                span: self.span(),
                first_searched,
                second: IncludeLibErrorVariant::NotFound {
                    searched: second_searched,
                },
            })
        } else {
            Err(DirectiveError::IncludeLibError {
                span: self.span(),
                first_searched,
                second: IncludeLibErrorVariant::NoAppNameComponent,
            })
        }
    }

    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for IncludeLib {}
impl PartialEq for IncludeLib {
    fn eq(&self, other: &Self) -> bool {
        self.path.symbol() == other.path.symbol()
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Error {}
impl PartialEq for Error {
    fn eq(&self, other: &Self) -> bool {
        self.message.symbol() == other.message.symbol()
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Warning {}
impl PartialEq for Warning {
    fn eq(&self, other: &Self) -> bool {
        self.message.symbol() == other.message.symbol()
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Endif {}
impl PartialEq for Endif {
    fn eq(&self, _other: &Self) -> bool {
        true
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Else {}
impl PartialEq for Else {
    fn eq(&self, _other: &Self) -> bool {
        true
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl Eq for Undef {}
impl PartialEq for Undef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for If {}
impl PartialEq for If {
    fn eq(&self, other: &Self) -> bool {
        self.condition == other.condition
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
}
impl Eq for Elif {}
impl PartialEq for Elif {
    fn eq(&self, other: &Self) -> bool {
        self.condition == other.condition
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl Eq for Ifdef {}
impl PartialEq for Ifdef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.2;
        SourceSpan::new(start, end)
    }
    pub fn name(&self) -> Symbol {
        self.name.symbol()
    }
}
impl Eq for Ifndef {}
impl PartialEq for Ifndef {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
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
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.0;
        SourceSpan::new(start, end)
    }
}
impl Eq for Define {}
impl PartialEq for Define {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
            && self.variables == other.variables
            && self.replacement == other.replacement
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
                replacement.push(reader.read_token()?);
                //match reader.read_token()? {
                //    token @ LexicalToken(_, Token::Dot, _) => {
                //        println!("yay {:?}", token);
                //        return Err(PreprocessorError::UnexpectedToken(token, Vec::new()));
                //    }
                //    token => replacement.push(token),
                //}
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct File {
    pub _hyphen: SymbolToken,
    pub _file: AtomToken,
    pub _open_paren: SymbolToken,
    pub path: StringToken,
    pub _comma: SymbolToken,
    pub line: IntegerToken,
    pub _close_paren: SymbolToken,
    pub _dot: SymbolToken,
}
impl File {
    pub fn span(&self) -> SourceSpan {
        let start = self._hyphen.0;
        let end = self._dot.0;
        SourceSpan::new(start, end)
    }
}
impl Eq for File {}
impl PartialEq for File {
    fn eq(&self, other: &Self) -> bool {
        self.path.symbol() == other.path.symbol() && self.line == other.line
    }
}
impl fmt::Display for File {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "-file({}, {}).", self.path, self.line,)
    }
}
impl ReadFrom for File {
    fn read_from<R, S>(reader: &mut R) -> Result<Self>
    where
        R: TokenReader<Source = S>,
    {
        Ok(File {
            _hyphen: reader.read_expected(&Token::Minus)?,
            _file: reader.read_expected(&symbols::File)?,
            _open_paren: reader.read_expected(&Token::LParen)?,
            path: reader.read()?,
            _comma: reader.read_expected(&Token::Comma)?,
            line: reader.read()?,
            _close_paren: reader.read_expected(&Token::RParen)?,
            _dot: reader.read_expected(&Token::Dot)?,
        })
    }
}

use std::env;
use std::fmt::Debug;
use std::path::{Path, PathBuf};

use firefly_util::diagnostics::{Diagnostic, Label, SourceIndex, SourceSpan, ToDiagnostic};

#[derive(Debug, thiserror::Error)]
pub enum PathVariableSubstituteError {
    #[error("invalid path substition variable {variable:?}")]
    InvalidPathVariable {
        variable: String,
        source: std::env::VarError,
    },
}
impl ToDiagnostic for PathVariableSubstituteError {
    fn to_diagnostic(self) -> Diagnostic {
        match self {
            PathVariableSubstituteError::InvalidPathVariable {
                source: env::VarError::NotPresent,
                variable,
            } => Diagnostic::error().with_message(format!(
                "invalid environment variable '{}': not defined",
                variable,
            )),
            PathVariableSubstituteError::InvalidPathVariable {
                source: env::VarError::NotUnicode { .. },
                variable,
            } => Diagnostic::error().with_message(format!(
                "invalid environment variable '{}': contains invalid unicode data",
                variable,
            )),
        }
    }
}

pub fn substitute_path_variables<P: AsRef<Path>>(
    path: P,
) -> Result<PathBuf, PathVariableSubstituteError> {
    let mut new = PathBuf::new();
    for c in path.as_ref().components() {
        if let Some(s) = c.as_os_str().to_str() {
            if s.as_bytes().get(0) == Some(&b'$') {
                let var = s.split_at(1).1;
                match env::var(var) {
                    Ok(c) => {
                        new.push(c);
                        continue;
                    }
                    Err(e) => {
                        return Err(PathVariableSubstituteError::InvalidPathVariable {
                            variable: var.to_owned(),
                            source: e,
                        });
                    }
                }
            }
        }
        new.push(c.as_os_str());
    }
    Ok(new)
}

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum EscapeStmError<D: Debug> {
    /// Unknown escape character
    #[error("unknown escape character '{escape_char}'")]
    UnknownEscape { range: (D, D), escape_char: char },
    /// Expected a base-n digit or alt
    #[error("expected base{base} digit or '{alt}', found '{found}'")]
    InvalidBaseNDigitOrAlt {
        range: (D, D),
        found: char,
        base: usize,
        alt: char,
    },
    /// Expected a base-n digit
    #[error("expected base{base} digit, found '{found}'")]
    InvalidBaseNDigit {
        range: (D, D),
        found: char,
        base: usize,
    },
    /// Expected control character symbol
    #[error("expected control character symbol (a-z), found '{found}'")]
    InvalidControl { range: (D, D), found: char },
    #[error("unexpected end of file")]
    UnexpectedEof { range: (D, D) },
}

impl<D: Debug> EscapeStmError<D> {
    pub fn range(&self) -> &(D, D) {
        match self {
            Self::UnknownEscape { range, .. } => range,
            Self::InvalidBaseNDigitOrAlt { range, .. } => range,
            Self::InvalidBaseNDigit { range, .. } => range,
            Self::InvalidControl { range, .. } => range,
            Self::UnexpectedEof { range } => range,
        }
    }
}

impl EscapeStmError<SourceIndex> {
    pub fn span(&self) -> SourceSpan {
        let (start, end) = self.range();
        SourceSpan::new(*start, *end)
    }
}

impl ToDiagnostic for EscapeStmError<SourceIndex> {
    fn to_diagnostic(self) -> Diagnostic {
        let msg = self.to_string();
        let span = self.span();

        Diagnostic::error()
            .with_message("invalid string escape")
            .with_labels(vec![
                Label::primary(span.source_id(), span).with_message(msg)
            ])
    }
}

/// Erlang string escape state machine.
#[derive(Clone, Debug)]
pub struct EscapeStm<D: Debug> {
    buf: String,
    curr_start: Option<D>,
    state: EscapeStmState,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EscapeStmState {
    Norm,
    Escape,
    Oct,
    HexStart,
    HexN,
    Hex2,
    Control,
}

#[derive(Debug)]
pub enum EscapeStmAction {
    Next,
    Again,
}

#[derive(Debug)]
pub struct EscapeStmOut<D: Debug> {
    pub range: (D, D),
    pub cp: u64,
}

impl<D: Copy + Debug> EscapeStm<D> {
    pub fn new() -> Self {
        EscapeStm {
            buf: String::new(),
            curr_start: None,
            state: EscapeStmState::Norm,
        }
    }

    pub fn reset(&mut self) {
        self.buf.clear();
        self.curr_start = None;
        self.state = EscapeStmState::Norm;
    }

    pub fn transition(
        &mut self,
        c: Option<char>,
        pos: D,
    ) -> Result<(EscapeStmAction, Option<EscapeStmOut<D>>), EscapeStmError<D>> {
        use EscapeStmAction as A;
        use EscapeStmState as S;

        let mut range = (self.curr_start.unwrap_or(pos), pos);
        let mut out = None;

        let action = match self.state {
            S::Norm => match c {
                Some('\\') => {
                    self.state = S::Escape;
                    self.curr_start = Some(pos);
                    A::Next
                }
                Some(c) => {
                    self.state = S::Norm;
                    range = (pos, pos);
                    out = Some(c as u64);
                    A::Next
                }
                None => A::Next,
            },
            S::Escape => {
                match c {
                    Some('b') => {
                        // Backspace
                        self.state = S::Norm;
                        out = Some('\x08' as u64);
                    }
                    Some('d') => {
                        // Delete
                        self.state = S::Norm;
                        out = Some('\x7f' as u64);
                    }
                    Some('e') => {
                        // Escape
                        self.state = S::Norm;
                        out = Some('\x1b' as u64);
                    }
                    Some('f') => {
                        // Form feed
                        self.state = S::Norm;
                        out = Some('\x0c' as u64);
                    }
                    Some('n') => {
                        // Line feed
                        self.state = S::Norm;
                        out = Some('\n' as u64);
                    }
                    Some('r') => {
                        // Carriage return
                        self.state = S::Norm;
                        out = Some('\r' as u64);
                    }
                    Some('s') => {
                        // Space
                        self.state = S::Norm;
                        out = Some(' ' as u64);
                    }
                    Some('t') => {
                        // Tab
                        self.state = S::Norm;
                        out = Some('\t' as u64);
                    }
                    Some('v') => {
                        // Vertical tab
                        self.state = S::Norm;
                        out = Some('\x0b' as u64);
                    }
                    Some(n) if n >= '0' && n <= '7' => {
                        self.buf.clear();
                        self.buf.push(n);
                        self.state = S::Oct;
                    }
                    Some('x') => {
                        self.state = S::HexStart;
                    }
                    Some('^') => {
                        self.state = S::Control;
                    }
                    Some(c) => {
                        self.state = S::Norm;
                        out = Some(c as u64);
                    }
                    None => return Err(EscapeStmError::UnexpectedEof { range }),
                }
                A::Next
            }
            S::Oct => match c {
                Some(c) if c >= '0' && c <= '7' => {
                    self.buf.push(c);

                    if self.buf.len() == 3 {
                        let parsed = u64::from_str_radix(&self.buf, 8).unwrap();

                        self.state = S::Norm;
                        out = Some(parsed);
                    } else {
                        self.state = S::Oct;
                    }

                    A::Next
                }
                _ => {
                    let parsed = u64::from_str_radix(&self.buf, 8).unwrap();

                    self.state = S::Norm;
                    out = Some(parsed);

                    A::Again
                }
            },
            S::HexStart => match c {
                Some('{') => {
                    self.state = S::HexN;

                    A::Next
                }
                Some(n) if n.is_digit(16) => {
                    self.buf.clear();
                    self.buf.push(n);
                    self.state = S::Hex2;

                    A::Next
                }
                Some(c) => {
                    return Err(EscapeStmError::InvalidBaseNDigitOrAlt {
                        range,
                        found: c,
                        base: 16,
                        alt: '{',
                    });
                }
                None => return Err(EscapeStmError::UnexpectedEof { range }),
            },
            S::Hex2 => match c {
                Some(n) if n.is_digit(16) => {
                    self.buf.push(n);
                    self.state = S::Norm;

                    let parsed = u64::from_str_radix(&self.buf, 16).unwrap();
                    out = Some(parsed);

                    A::Next
                }
                Some(c) => {
                    return Err(EscapeStmError::InvalidBaseNDigit {
                        range,
                        found: c,
                        base: 16,
                    });
                }
                None => return Err(EscapeStmError::UnexpectedEof { range }),
            },
            S::HexN => match c {
                Some('}') => {
                    let parsed = u64::from_str_radix(&self.buf, 16).unwrap();

                    self.state = S::Norm;
                    out = Some(parsed);

                    A::Next
                }
                Some(n) if n.is_digit(16) => {
                    self.buf.push(n);
                    self.state = S::HexN;

                    A::Next
                }
                Some(c) => {
                    return Err(EscapeStmError::InvalidBaseNDigitOrAlt {
                        range,
                        found: c,
                        base: 16,
                        alt: '}',
                    });
                }
                None => return Err(EscapeStmError::UnexpectedEof { range }),
            },
            S::Control => {
                match c {
                    Some(c) => {
                        let num = (c as u64) % 32;
                        self.state = S::Norm;
                        out = Some(num);

                        if c < '@' || c > '~' {
                            // TODO: Warn?
                            // return Err(EscapeStmError::InvalidControl { range, found: c })
                        }

                        A::Next
                    }
                    None => return Err(EscapeStmError::UnexpectedEof { range }),
                }
            }
        };

        Ok((action, out.map(|c| EscapeStmOut { cp: c, range })))
    }
}

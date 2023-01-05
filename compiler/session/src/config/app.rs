///! This module provides a struct that contains metadata about an Erlang application.
///!
///! It implements a limited parser for Erlang application resource files - i.e. `foo.app`
///! or `foo.app.src` - sufficient to provide us with the key details about an Erlang app.
use std::fmt;
use std::ops::{Deref, Range};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail};
use firefly_intern::Symbol;
use logos::Logos;

/// Metadata about an Erlang application
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct App {
    /// The name of the application
    pub name: Symbol,
    /// The specified version of the application. Not required.
    pub version: Option<String>,
    /// The root directory in which the application was found. Not required.
    pub root: Option<PathBuf>,
    /// The set of modules names contained in this application.
    ///
    /// NOTE: This may not match the full set of modules found on disk, but
    /// is the set of modules which systools would package in a release, so
    /// if they don't match, it is likely a mistake.
    pub modules: Vec<Symbol>,
    /// The full set of applications this application depends on.
    pub applications: Vec<Symbol>,
    /// For OTP applications (i.e. those with a supervisor tree), this is the
    /// application callback module for this application. If not present, then
    /// this is just a library application
    pub otp_module: Option<Symbol>,
}
impl App {
    /// Create a new empty application with the given name
    pub fn new(name: Symbol) -> Self {
        Self {
            name,
            version: None,
            root: None,
            modules: vec![],
            applications: vec![],
            otp_module: None,
        }
    }

    /// Parse an application resource from the given path
    pub fn parse<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let path = path.as_ref();
        let root = path.parent().unwrap().parent().unwrap().to_path_buf();
        let source = std::fs::read_to_string(path)?;
        parse_app(&source).map(|mut app| {
            app.root.replace(root);
            app
        })
    }

    /// Parse an application resource from the given string
    ///
    /// NOTE: The resulting manifest will not have `root` set, make sure
    /// you set it manually if the application has a corresponding root
    /// directory
    pub fn parse_str<S: AsRef<str>>(source: S) -> anyhow::Result<Self> {
        parse_app(source)
    }
}

#[derive(Logos, Copy, Clone, Debug, PartialEq)]
enum Token {
    // Punctuation
    #[token("{")]
    Lbrace,
    #[token("}")]
    Rbrace,
    #[token("[")]
    Lbracket,
    #[token("]")]
    Rbracket,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,

    // Comments
    #[regex(r"%[^\n]*", logos::skip)]
    Comment,

    // Literals
    #[regex(r"[a-z][a-zA-Z_0-9]+")]
    Atom,
    #[regex(r"[0-9]+")]
    Integer,
    #[regex(r#""([^"\\]|\\t|\\u|\\n|\\")*""#)]
    String,

    #[error]
    #[regex(r"[ \t\n\f ]+", logos::skip)]
    Error,
}
impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;
        match self {
            Self::Lbrace => f.write_char('{'),
            Self::Rbrace => f.write_char('}'),
            Self::Lbracket => f.write_char('['),
            Self::Rbracket => f.write_char(']'),
            Self::Dot => f.write_char('.'),
            Self::Comma => f.write_char(','),
            Self::Comment => f.write_str("COMMENT"),
            Self::Atom => f.write_str("ATOM"),
            Self::Integer => f.write_str("INTEGER"),
            Self::String => f.write_str("STRING"),
            Self::Error => f.write_str("ERROR"),
        }
    }
}

struct Lexer<'a> {
    lex: logos::Lexer<'a, Token>,
    curr: Token,
    span: Range<usize>,
    lines: Vec<Range<usize>>,
}
impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        // Get a mapping of character ranges to lines
        let lines = {
            let mut lines = Vec::<Range<usize>>::with_capacity(10);
            let mut line_start = 0;
            let mut line_end = 0;
            for (idx, c) in source.char_indices() {
                if c == '\n' {
                    lines.push(Range {
                        start: line_start,
                        end: line_end,
                    });
                    line_start = idx + 1;
                    line_end = 0;
                } else {
                    line_end = idx;
                }
            }
            // Last line has to be pushed outside the loop
            lines.push(Range {
                start: line_start,
                end: line_end,
            });
            lines
        };

        Self {
            lex: Token::lexer(source),
            curr: Token::Error,
            span: 0..0,
            lines,
        }
    }

    fn span(&self) -> Range<usize> {
        self.lex.span()
    }

    fn slice(&self) -> &str {
        self.lex.slice()
    }

    fn current_token(&self) -> Token {
        self.curr
    }

    fn current_location(&self) -> Location {
        self.span_to_loc(self.span.clone())
    }

    fn span_to_loc(&self, span: Range<usize>) -> Location {
        let start_index = span.start;
        let loc = self.lines.iter().enumerate().find_map(|(i, line)| {
            if start_index <= line.end {
                Some(Location(i + 1, (line.end - start_index) + 1))
            } else {
                None
            }
        });
        match loc {
            None => panic!("expected to find loc for span {:?}", span),
            Some(loc) => loc,
        }
    }
}
impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Token> {
        if let Some(next) = self.lex.next() {
            let span = self.lex.span();
            self.curr = next;
            self.span = span;
            return Some(self.curr);
        }

        None
    }
}

#[derive(Clone)]
struct Spanned<T> {
    item: T,
    span: Range<usize>,
}
impl<T> Spanned<T> {
    fn new(span: Range<usize>, item: T) -> Self {
        Self { item, span }
    }
}
impl<T> Deref for Spanned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.item
    }
}

#[derive(Clone)]
enum Term {
    Atom(Symbol),
    Integer(i64),
    String(String),
    Tuple(Tuple),
    List(List),
}
impl Term {
    fn as_atom(self) -> anyhow::Result<Symbol> {
        match self {
            Self::Atom(a) => Ok(a),
            other => bail!("expected atom, but got '{}'", &other),
        }
    }
    fn as_string(self) -> anyhow::Result<String> {
        self.try_into()
    }
    fn as_tuple(self) -> anyhow::Result<Tuple> {
        Tuple::try_from(self)
    }
    fn as_list(self) -> anyhow::Result<List> {
        List::try_from(self)
    }
}
impl TryInto<i64> for Term {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<i64, Self::Error> {
        match self {
            Self::Integer(i) => Ok(i),
            other => Err(anyhow!("expected integer, but got '{}'", &other)),
        }
    }
}
impl TryInto<String> for Term {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<String, Self::Error> {
        match self {
            Self::String(s) => Ok(s),
            other => Err(anyhow!("expected string, but got '{}'", &other)),
        }
    }
}
impl fmt::Display for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use std::fmt::Write;
        match self {
            Self::Atom(a) => write!(f, "{}", a.as_str()),
            Self::Integer(i) => write!(f, "{}", i),
            Self::String(s) => write!(f, "\"{}\"", s),
            Self::List(List(terms)) | Self::Tuple(Tuple(terms)) => {
                f.write_char('{')?;
                for (i, t) in terms.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{}", &t.item)?;
                }
                f.write_char('}')
            }
        }
    }
}

#[derive(Clone)]
struct Tuple(Vec<Spanned<Term>>);
impl Tuple {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get(&self, index: usize) -> Option<Spanned<Term>> {
        self.0.get(index).map(|t| t.clone())
    }
}
impl TryFrom<Term> for Tuple {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::Tuple(tuple) => Ok(tuple),
            other => Err(anyhow!("expected tuple, but got '{}'", &other)),
        }
    }
}

#[derive(Clone)]
struct List(Vec<Spanned<Term>>);
impl List {
    fn drain(&mut self) -> std::vec::Drain<'_, Spanned<Term>> {
        self.0.drain(0..)
    }
}
impl TryFrom<Term> for List {
    type Error = anyhow::Error;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        match term {
            Term::List(list) => Ok(list),
            other => Err(anyhow!("expected list, but got '{}'", &other)),
        }
    }
}

#[derive(Copy, Clone)]
struct Location(usize, usize);
impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.0, self.1)
    }
}

fn parse_app<S: AsRef<str>>(source: S) -> anyhow::Result<App> {
    let source = source.as_ref();
    let mut lex = Lexer::new(source);

    // Make sure we have a minimum viable spec
    let mut contents = parse_root(&mut lex)?;
    let resource = contents.pop().map(|r| r.item).unwrap().as_tuple()?;
    if resource.len() < 3 {
        bail!("invalid resource, application spec must be a tuple of 3 elements");
    }
    // We expect the tuple to be tagged 'applciation'
    {
        let tag = resource.get(0).unwrap();
        let span = tag.span;
        if tag.item.as_atom()? != "application" {
            bail!("expected atom 'application' at {}", lex.span_to_loc(span));
        }
    }
    // We expect an atom as the second element in the tuple
    let name = {
        let name = resource.get(1).unwrap();
        let span = name.span;
        name.item
            .as_atom()
            .map_err(|e| anyhow!("{} at {}", e, lex.span_to_loc(span)))?
    };

    // Initialize default app metadata with the parsed name
    let mut app = App::new(name);

    // We expect the third element to be a (possibly empty) list
    let mut meta = {
        let meta = resource.get(2).unwrap();
        let span = meta.span;
        meta.item
            .as_list()
            .map_err(|e| anyhow!("{} at {}", e, lex.span_to_loc(span)))?
    };

    // Iterate over the application metadata, handling keys we are interested in
    for item in meta.drain() {
        let span = item.span;
        let item = item.item.as_tuple()?;
        // Must be a valid keyword item
        if item.len() != 2 {
            bail!("invalid keyword list item at {}", lex.span_to_loc(span));
        }
        // Keys must be atoms
        let key = {
            let key = item.get(0).unwrap();
            let span = key.span;
            key.item
                .as_atom()
                .map_err(|e| anyhow!("{} at {}", e, lex.span_to_loc(span)))?
        };
        let value = item.get(1).unwrap();
        let value_span = value.span;
        let value = value.item;
        match key.as_str().get() {
            "vsn" => {
                app.version.replace(value.as_string()?);
            }
            "modules" => {
                let mut modules = value.as_list()?;
                for module in modules.drain().map(|m| m.item) {
                    app.modules.push(module.as_atom()?);
                }
            }
            "applications" => {
                let mut applications = value.as_list()?;
                for application in applications.drain().map(|a| a.item) {
                    app.applications.push(application.as_atom()?);
                }
            }
            "mod" => {
                let mod_tuple = value.as_tuple()?;
                if mod_tuple.len() != 2 {
                    bail!(
                        "invalid value for 'mod' at {}, tuple must contain 2 elements",
                        lex.span_to_loc(value_span)
                    );
                }
                let module_name = mod_tuple.get(0).map(|t| t.item).unwrap().as_atom()?;
                app.otp_module.replace(module_name);
            }
            _ => continue,
        }
    }

    Ok(app)
}

/// Parses the root content of a resource file
///
/// A resource file can contain comments, and one or more terms, each terminated with '.'
///
/// An application resource file is a special case though, in that it should only contain a single
/// item, but we let the caller handle that
fn parse_root(lexer: &mut Lexer<'_>) -> anyhow::Result<Vec<Spanned<Term>>> {
    let mut contents = Vec::with_capacity(1);
    loop {
        let item = parse_term(lexer);
        if item.is_none() {
            if contents.is_empty() {
                bail!("expected term, but got eof");
            }
            return Ok(contents);
        }
        match item.unwrap()? {
            Ok(term) => {
                contents.push(term);
                let loc = lexer.current_location();
                let next = lexer.next();
                if next.is_none() {
                    bail!(
                        "expected '.' to follow term starting at {}, but got eof",
                        loc
                    );
                }
                match next.unwrap() {
                    Token::Dot => continue,
                    token => {
                        let loc = lexer.current_location();
                        bail!("expected '.' at {}, but got '{}'", loc, token);
                    }
                }
            }
            Err(token) => {
                let loc = lexer.current_location();
                bail!("expected term at {}, but got '{}'", loc, token);
            }
        }
    }
}

fn parse_term(lexer: &mut Lexer<'_>) -> Option<anyhow::Result<Result<Spanned<Term>, Token>>> {
    let next = lexer.next();
    if next.is_none() {
        return None;
    }
    match next.unwrap() {
        Token::Lbrace => {
            let span = lexer.span();
            match parse_terms(lexer) {
                Ok(terms) => Some(Ok(Ok(Spanned::new(span, Term::Tuple(Tuple(terms)))))),
                Err(err) => Some(Err(err)),
            }
        }
        Token::Lbracket => {
            let span = lexer.span();
            match parse_terms(lexer) {
                Ok(terms) => Some(Ok(Ok(Spanned::new(span, Term::List(List(terms)))))),
                Err(err) => Some(Err(err)),
            }
        }
        Token::Atom => {
            let span = lexer.span();
            let value = Symbol::intern(lexer.slice());
            Some(Ok(Ok(Spanned::new(span, Term::Atom(value)))))
        }
        Token::String => {
            let span = lexer.span();
            let value = lexer.slice();
            // Trim quotes
            let len = value.len();
            let value = &value[1..(len - 1)];
            // Unescape contents
            let unescaped = value.chars().filter(|c| *c != '\\').collect();
            Some(Ok(Ok(Spanned::new(span, Term::String(unescaped)))))
        }
        Token::Integer => {
            let span = lexer.span();
            let value = match lexer.slice().parse() {
                Ok(i) => i,
                Err(e) => return Some(Err(anyhow!("{}", e))),
            };
            Some(Ok(Ok(Spanned::new(span, Term::Integer(value)))))
        }
        token => Some(Ok(Err(token))),
    }
}

fn parse_terms(lexer: &mut Lexer<'_>) -> anyhow::Result<Vec<Spanned<Term>>> {
    let terminator = match lexer.current_token() {
        Token::Lbrace => '}',
        Token::Lbracket => ']',
        _ => panic!("invalid call to parse_terms"),
    };
    let mut terms = Vec::with_capacity(2);
    loop {
        // Handle early end of input
        let result = parse_term(lexer);
        if result.is_none() {
            bail!("expected ',' or '{}', got eof", terminator);
        }
        // If an error occurred, propagate it upwards
        let result = result.unwrap()?;
        match result {
            Ok(term) => {
                terms.push(term);
            }
            // Handle empty sequence
            Err(Token::Rbrace) if terminator == '}' => {
                return Ok(terms);
            }
            Err(Token::Rbracket) if terminator == ']' => {
                return Ok(terms);
            }
            // All other tokens are syntax errors
            Err(_token) => {
                let loc = lexer.current_location();
                let invalid = lexer.slice();
                bail!(
                    "invalid syntax at {}, expected term, got '{}'",
                    loc,
                    invalid
                );
            }
        }
        // Check for next item/end of sequence
        let next = lexer.next();
        if next.is_none() {
            bail!("expected ',' or '{}', got eof", terminator);
        }
        match next.unwrap() {
            Token::Rbrace if terminator == '}' => {
                return Ok(terms);
            }
            Token::Rbracket if terminator == ']' => {
                return Ok(terms);
            }
            Token::Comma => continue,
            _token => {
                let loc = lexer.current_location();
                let invalid = lexer.slice();
                bail!(
                    "invalid syntax at {}, expected ',' or '{}', got '{}'",
                    loc,
                    terminator,
                    invalid
                );
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    const SIMPLE: &'static str = "{application, simple, []}.";
    const RICH: &'static str = r#"
%% A multi-line
%% comment
{application, example,
  [{description, "An example application"},
   {vsn, "0.1.0-rc0"},
   %% Another comment
   {modules, [example_app, example_sup, example_worker]},
   {registered, [example_registry]},
   {applications, [kernel, stdlib, sasl]},
   {mod, {example_app, []}}
  ]}.
"#;
    const NOT_EVEN_A_RESOURCE: &'static str = r#""hi"."#;
    const INVALID_APP_RESOURCE: &'static str = "{}.";
    const MISSING_TRAILING_DOT: &'static str = "{application, simple, []}";
    const UNRECOGNIZED_SYNTAX: &'static str =
        "{application, simple, [{mod, {simple_app, [#{foo => bar}]}}]}";

    #[test]
    fn simple_app_resource_test() {
        let app = App::parse_str(SIMPLE).unwrap();
        let name = app.name.as_str().get();
        assert_eq!(name, "simple");
    }

    #[test]
    fn rich_app_resource_test() {
        let app = App::parse_str(RICH).unwrap();
        let name = app.name.as_str().get();
        assert_eq!(name, "example");
        assert_eq!(app.version.as_ref().map(|s| s.as_str()), Some("0.1.0-rc0"));
        assert_eq!(app.modules.len(), 3);
        assert_eq!(app.applications.len(), 3);
        assert_eq!(
            app.otp_module.map(|s| s.as_str().get()),
            Some("example_app")
        );
    }

    #[test]
    #[should_panic(expected = "expected tuple, but got '\"hi\"'")]
    fn invalid_manifest_not_even_a_resource() {
        App::parse_str(NOT_EVEN_A_RESOURCE).unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid resource, application spec must be a tuple of 3 elements")]
    fn invalid_manifest_invalid_resource() {
        App::parse_str(INVALID_APP_RESOURCE).unwrap();
    }

    #[test]
    #[should_panic(expected = "expected '.' to follow term starting at 1:1, but got eof")]
    fn invalid_manifest_missing_trailing_dot() {
        App::parse_str(MISSING_TRAILING_DOT).unwrap();
    }

    #[test]
    #[should_panic(expected = "invalid syntax at 1:18, expected term, got '#'")]
    fn invalid_manifest_unrecognized_syntax() {
        App::parse_str(UNRECOGNIZED_SYNTAX).unwrap();
    }
}

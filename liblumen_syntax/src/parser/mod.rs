/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => {
        ByteSpan::new($l, $r)
    };
    ($i:expr) => {
        ByteSpan::new($i, $i)
    };
}

/// Convenience function for building parser errors
macro_rules! to_lalrpop_err (
    ($error:expr) => (lalrpop_util::ParseError::User { error: $error })
);

#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
pub(crate) mod grammar {
    // During the build step, `build.rs` will output the generated parser to `OUT_DIR` to avoid
    // adding it to the source directory, so we just directly include the generated parser here.
    //
    // Even with `.gitignore` and the `exclude` in the `Cargo.toml`, the generated parser can still
    // end up in the source directory. This could happen when `cargo build` builds the file out of
    // the Cargo cache (`$HOME/.cargo/registrysrc`), and the build script would then put its output
    // in that cached source directory because of https://github.com/lalrpop/lalrpop/issues/280.
    // Later runs of `cargo vendor` then copy the source from that directory, including the
    // generated file.
    include!(concat!(env!("OUT_DIR"), "/parser/grammar.rs"));
}

#[macro_use] mod macros;

/// Contains the visitor trait needed to traverse the AST and helper walk functions.
pub mod visitor;
pub mod ast;
mod errors;

use std::borrow::Cow;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use liblumen_diagnostics::{CodeMap, FileName};

use crate::lexer::{FileMapSource, Lexer, Scanner, Source, Symbol};
use crate::preprocessor::{MacroDef, Preprocessed, Preprocessor};

pub use self::errors::*;

/// The type of result returned from parsing functions
pub type ParseResult<T> = Result<T, Vec<ParserError>>;

pub struct Parser {
    pub config: ParseConfig,
}
impl Parser {
    pub fn new(config: ParseConfig) -> Parser {
        Parser { config }
    }

    pub fn parse_string<S, T>(&self, source: S) -> ParseResult<T>
    where
        S: AsRef<str>,
        T: Parse,
    {
        let filemap = {
            self.config.codemap.lock().unwrap().add_filemap(
                FileName::Virtual(Cow::Borrowed("nofile")),
                source.as_ref().to_owned(),
            )
        };
        <T as Parse<T>>::parse(&self.config, FileMapSource::new(filemap))
    }

    pub fn parse_file<P, T>(&self, path: P) -> ParseResult<T>
    where
        P: AsRef<Path>,
        T: Parse,
    {
        match FileMapSource::from_path(self.config.codemap.clone(), path) {
            Err(err) => return Err(vec![err.into()]),
            Ok(source) => <T as Parse<T>>::parse(&self.config, source),
        }
    }
}

pub struct ParseConfig {
    pub codemap: Arc<Mutex<CodeMap>>,
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub code_paths: VecDeque<PathBuf>,
    pub macros: Option<HashMap<Symbol, MacroDef>>,
}
impl ParseConfig {
    pub fn new(codemap: Arc<Mutex<CodeMap>>) -> Self {
        ParseConfig {
            codemap,
            warnings_as_errors: false,
            no_warn: false,
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}
impl Default for ParseConfig {
    fn default() -> Self {
        ParseConfig {
            codemap: Arc::new(Mutex::new(CodeMap::new())),
            warnings_as_errors: false,
            no_warn: false,
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}

pub trait Parse<T = Self> {
    type Parser;

    /// Initializes a token stream for the underlying parser and invokes parse_tokens/1
    fn parse<S>(config: &ParseConfig, source: S) -> ParseResult<T>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(config, lexer);
        Self::parse_tokens(tokens)
    }

    /// Implemented by each parser, which should parse the token stream and produce a T
    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<T>;
}

impl Parse for ast::Module {
    type Parser = grammar::ModuleParser;

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<ast::Module> {
        let mut errs = Vec::new();
        let result = Self::Parser::new()
            .parse(&mut errs, tokens)
            .map_err(|e| e.map_error(|ei| ei.into()));
        to_parse_result(errs, result)
    }
}

impl Parse for ast::Expr {
    type Parser = grammar::ExprParser;

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(tokens: S) -> ParseResult<ast::Expr> {
        let mut errs = Vec::new();
        let result = Self::Parser::new()
            .parse(&mut errs, tokens)
            .map_err(|e| e.map_error(|ei| ei.into()));
        to_parse_result(errs, result)
    }
}

fn to_parse_result<T>(mut errs: Vec<ParseError>, result: Result<T, ParseError>) -> ParseResult<T> {
    match result {
        Ok(ast) => {
            if errs.len() > 0 {
                return Err(errs.drain(0..).map(ParserError::from).collect());
            }
            Ok(ast)
        }
        Err(err) => {
            errs.push(err);
            Err(errs.drain(0..).map(ParserError::from).collect())
        }
    }
}

#[cfg(test)]
mod test {
    use pretty_assertions::assert_eq;

    use super::ast::*;
    use super::*;

    use liblumen_diagnostics::ByteSpan;
    use liblumen_diagnostics::{ColorChoice, Emitter, StandardStreamEmitter};

    use crate::lexer::{Ident, Symbol};
    use crate::preprocessor::PreprocessorError;

    fn parse<T>(input: &'static str) -> T
    where
        T: Parse<T>,
    {
        let config = ParseConfig::default();
        let parser = Parser::new(config);
        let errs = match parser.parse_string::<&'static str, T>(input) {
            Ok(ast) => return ast,
            Err(errs) => errs,
        };
        let emitter = StandardStreamEmitter::new(ColorChoice::Auto)
            .set_codemap(parser.config.codemap.clone());
        for err in errs.iter() {
            emitter.diagnostic(&err.to_diagnostic()).unwrap();
        }
        panic!("parse failed");
    }

    fn parse_fail<T>(input: &'static str) -> Vec<ParserError>
    where
        T: Parse<T>,
    {
        let config = ParseConfig::default();
        let parser = Parser::new(config);
        match parser.parse_string::<&'static str, T>(input) {
            Err(errs) => errs,
            _ => panic!("expected parse to fail, but it succeeded!"),
        }
    }

    macro_rules! module {
        ($name:expr, $body:expr) => {
            {
                let mut errs = Vec::new();
                let module = Module::new(&mut errs, ByteSpan::default(), $name, $body);
                if errs.len() > 0 {
                    let emitter = StandardStreamEmitter::new(ColorChoice::Auto);
                    for err in errs.drain(..) {
                        let err = ParserError::from(err);
                        emitter.diagnostic(&err.to_diagnostic()).unwrap();
                    }
                    panic!("failed to create expected module!");
                }
                module
            }
        }
    }

    #[test]
    fn parse_empty_module() {
        let result: Module = parse("-module(foo).");
        let expected = module!(ident!("foo"), vec![]);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_module_with_multi_clause_function() {
        let result: Module = parse(
            "-module(foo).

foo([], Acc) -> Acc;
foo([H|T], Acc) -> foo(T, [H|Acc]).
",
        );
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(foo),
            params: vec![nil!(), var!(Acc)],
            guard: None,
            body: vec![var!(Acc)],
        });
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(foo),
            params: vec![cons!(var!(H), var!(T)), var!(Acc)],
            guard: None,
            body: vec![apply!(atom!(foo), var!(T), cons!(var!(H), var!(Acc)))],
        });
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: ByteSpan::default(),
            name: ident!("foo"),
            arity: 2,
            clauses,
            spec: None,
        }));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_if_expressions() {
        let result: Module = parse(
            "-module(foo).

unless(false) ->
    true;
unless(true) ->
    false;
unless(Value) ->
    if
        Value == 0 -> true;
        Value -> false;
        else -> true
    end.

",
        );
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(unless),
            params: vec![atom!(false)],
            guard: None,
            body: vec![atom!(true)],
        });
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(unless),
            params: vec![atom!(true)],
            guard: None,
            body: vec![atom!(false)],
        });
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(unless),
            params: vec![var!(Value)],
            guard: None,
            body: vec![Expr::If(If {
                span: ByteSpan::default(),
                clauses: vec![
                    IfClause {
                        span: ByteSpan::default(),
                        conditions: vec![Expr::BinaryExpr(BinaryExpr {
                            span: ByteSpan::default(),
                            lhs: Box::new(var!(Value)),
                            op: BinaryOp::Equal,
                            rhs: Box::new(int!(0)),
                        })],
                        body: vec![atom!(true)],
                    },
                    IfClause {
                        span: ByteSpan::default(),
                        conditions: vec![var!(Value)],
                        body: vec![atom!(false)],
                    },
                    IfClause {
                        span: ByteSpan::default(),
                        conditions: vec![atom!(else)],
                        body: vec![atom!(true)],
                    },
                ],
            })],
        });
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: ByteSpan::default(),
            name: ident!(unless),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_case_expressions() {
        let result: Module = parse(
            "-module(foo).

typeof(Value) ->
    case Value of
        [] -> nil;
        [_|_] -> list;
        N when is_number(N) -> N;
        _ -> other
    end.

",
        );
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(typeof),
            params: vec![var!(Value)],
            guard: None,
            body: vec![Expr::Case(Case {
                span: ByteSpan::default(),
                expr: Box::new(var!(Value)),
                clauses: vec![
                    Clause {
                        span: ByteSpan::default(),
                        pattern: nil!(),
                        guard: None,
                        body: vec![atom!(nil)],
                    },
                    Clause {
                        span: ByteSpan::default(),
                        pattern: cons!(var!(_), var!(_)),
                        guard: None,
                        body: vec![atom!(list)],
                    },
                    Clause {
                        span: ByteSpan::default(),
                        pattern: var!(N),
                        guard: Some(vec![Guard {
                            span: ByteSpan::default(),
                            conditions: vec![apply!(atom!(is_number), var!(N))],
                        }]),
                        body: vec![var!(N)],
                    },
                    Clause {
                        span: ByteSpan::default(),
                        pattern: var!(_),
                        guard: None,
                        body: vec![atom!(other)],
                    },
                ],
            })],
        });
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: ByteSpan::default(),
            name: ident!(typeof),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_receive_expressions() {
        let result: Module = parse(
            "-module(foo).

loop(State, Timeout) ->
    receive
        {From, {Ref, Msg}} ->
            From ! {Ref, ok},
            handle_info(Msg, State);
        _ ->
            exit(io_lib:format(\"unexpected message: ~p~n\", [Msg]))
    after
        Timeout ->
            timeout
    end.
",
        );
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(loop),
            params: vec![var!(State), var!(Timeout)],
            guard: None,
            body: vec![Expr::Receive(Receive {
                span: ByteSpan::default(),
                clauses: Some(vec![
                    Clause {
                        span: ByteSpan::default(),
                        pattern: tuple!(var!(From), tuple!(var!(Ref), var!(Msg))),
                        guard: None,
                        body: vec![
                            Expr::BinaryExpr(BinaryExpr {
                                span: ByteSpan::default(),
                                lhs: Box::new(var!(From)),
                                op: BinaryOp::Send,
                                rhs: Box::new(tuple!(var!(Ref), atom!(ok))),
                            }),
                            apply!(atom!(handle_info), var!(Msg), var!(State)),
                        ],
                    },
                    Clause {
                        span: ByteSpan::default(),
                        pattern: var!(_),
                        guard: None,
                        body: vec![
                            apply!(atom!(exit),
                                   apply!(remote!(io_lib, format),
                                          Expr::Literal(Literal::String(ident!("unexpected message: ~p~n"))),
                                          cons!(var!(Msg), nil!())))
                        ],
                    },
                ]),
                after: Some(After {
                    span: ByteSpan::default(),
                    timeout: Box::new(var!(Timeout)),
                    body: vec![atom!(timeout)],
                }),
            })],
        });
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: ByteSpan::default(),
            name: ident!(loop),
            arity: 2,
            clauses,
            spec: None,
        }));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_if() {
        let result: Module = parse(
            "-module(foo).
-define(TEST, true).
-define(OTP_VERSION, 21).

-ifdef(TEST).
env() ->
    test.
-else.
env() ->
    release.
-endif.

-if(?OTP_VERSION > 21).
system_version() ->
    future.
-elif(?OTP_VERSION == 21).
system_version() ->
    ?OTP_VERSION.
-else.
system_version() ->
    old.
-endif.
",
        );
        let mut body = Vec::new();
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(env),
            params: vec![],
            guard: None,
            body: vec![atom!(test)],
        });
        let env_fun = NamedFunction {
            span: ByteSpan::default(),
            name: ident!(env),
            arity: 0,
            clauses,
            spec: None,
        };
        body.push(TopLevel::Function(env_fun));

        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(system_version),
            params: vec![],
            guard: None,
            body: vec![int!(21)],
        });
        let system_version_fun = NamedFunction {
            span: ByteSpan::default(),
            name: ident!(system_version),
            arity: 0,
            clauses,
            spec: None,
        };
        body.push(TopLevel::Function(system_version_fun));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_warning_error() {
        // NOTE: Warnings are not printed with cfg(test), as we
        // cannot control where they end up without refactoring to pass
        // a writer everywhere. You can change this for testing by
        // going to the Preprocessor and finding the line where we handle
        // the warning directive and toggle the config flag
        let mut errs = parse_fail::<Module>(
            "-module(foo).
-warning(\"this is a compiler warning\").
-error(\"this is a compiler error\").
",
        );
        match errs.pop() {
            Some(ParserError::Preprocessor(PreprocessorError::CompilerError(_, _))) => (),
            Some(err) => panic!(
                "expected compiler error, but got a different error instead: {:?}",
                err
            ),
            None => panic!("expected compiler error, but didn't get any errors!"),
        }
    }

    #[test]
    fn parse_try() {
        let result: Module = parse(
            "-module(foo).

example(File) ->
    try read(File) of
        {ok, Contents} ->
            {ok, Contents}
    catch
        error:{Mod, Code} ->
            {error, Mod:format_error(Code)};
        Reason ->
            {error, Reason}
    after
        close(File)
    end.
",
        );
        let mut clauses = Vec::new();
        clauses.push(FunctionClause {
            span: ByteSpan::default(),
            name: ident_opt!(example),
            params: vec![var!(File)],
            guard: None,
            body: vec![Expr::Try(Try {
                span: ByteSpan::default(),
                exprs: Some(vec![apply!(atom!(read), var!(File))]),
                clauses: Some(vec![Clause {
                    span: ByteSpan::default(),
                    pattern: tuple!(atom!(ok), var!(Contents)),
                    guard: None,
                    body: vec![tuple!(atom!(ok), var!(Contents))],
                }]),
                catch_clauses: Some(vec![
                    TryClause {
                        span: ByteSpan::default(),
                        kind: Name::Atom(ident!(error)),
                        error: tuple!(var!(Mod), var!(Code)),
                        trace: ident!(_),
                        guard: None,
                        body: vec![tuple!(atom!(error), apply!(remote!(var!(Mod), atom!(format_error)), var!(Code)))],
                    },
                    TryClause {
                        span: ByteSpan::default(),
                        kind: Name::Atom(ident!(throw)),
                        error: var!(Reason),
                        trace: ident!(_),
                        guard: None,
                        body: vec![tuple!(atom!(error), var!(Reason))],
                    },
                ]),
                after: Some(vec![apply!(atom!(close), var!(File))]),
            })],
        });
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: ByteSpan::default(),
            name: ident!(example),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(ident!(foo), body);
        assert_eq!(result, expected);
    }
}

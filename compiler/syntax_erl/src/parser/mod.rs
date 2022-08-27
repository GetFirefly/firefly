/// Used in the grammar for easy span creation
macro_rules! span {
    ($l:expr, $r:expr) => {
        SourceSpan::new($l, $r)
    };
    ($i:expr) => {
        SourceSpan::new($i, $i)
    };
}

/// Convenience function for building parser errors
#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unknown_lints)]
#[allow(clippy)]
#[allow(unused_parens)]
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

pub mod binary;
mod errors;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use firefly_diagnostics::{CodeMap, Diagnostic, Reporter};
use firefly_intern::Symbol;
use firefly_parser::{Parse as GParse, Parser as GParser};
use firefly_parser::{Scanner, Source};

pub type Parser = GParser<ParseConfig>;
pub trait Parse<T> = GParse<T, Config = ParseConfig, Error = ParserError>;

use crate::ast;
use crate::lexer::Lexer;
use crate::preprocessor::{MacroContainer, MacroDef, MacroIdent, Preprocessed, Preprocessor};

pub use self::errors::*;

/// The type of result returned from parsing functions
pub type ParseResult<T> = Result<T, Vec<ParserError>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseConfig {
    pub warnings_as_errors: bool,
    pub no_warn: bool,
    pub include_paths: VecDeque<PathBuf>,
    pub code_paths: VecDeque<PathBuf>,
    pub macros: Option<MacroContainer>,
}
impl ParseConfig {
    pub fn new() -> Self {
        ParseConfig::default()
    }

    pub fn define<V: Into<MacroDef>>(&mut self, name: Symbol, value: V) {
        let macros = self.macros.get_or_insert_with(|| MacroContainer::new());
        macros.insert(MacroIdent::Const(name), value.into());
    }
}
impl Default for ParseConfig {
    fn default() -> Self {
        ParseConfig {
            warnings_as_errors: false,
            no_warn: false,
            include_paths: VecDeque::new(),
            code_paths: VecDeque::new(),
            macros: None,
        }
    }
}

impl GParse for ast::Module {
    type Parser = grammar::ModuleParser;
    type Error = ParserError;
    type Config = ParseConfig;
    type Token = Preprocessed;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(parser: &Parser, reporter: Reporter, source: S) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(parser, lexer, reporter.clone());
        Self::parse_tokens(reporter, parser.codemap.clone(), tokens)
    }

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(
        reporter: Reporter,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, Self::Error> {
        let result = Self::Parser::new().parse(&reporter, &codemap, tokens);
        to_parse_result(reporter, result)
    }
}

impl GParse for ast::Expr {
    type Parser = grammar::ExprParser;
    type Error = ParserError;
    type Config = ParseConfig;
    type Token = Preprocessed;

    fn root_file_error(source: std::io::Error, path: std::path::PathBuf) -> Self::Error {
        ParserError::RootFile { source, path }
    }

    fn parse<S>(parser: &Parser, reporter: Reporter, source: S) -> Result<Self, Self::Error>
    where
        S: Source,
    {
        let scanner = Scanner::new(source);
        let lexer = Lexer::new(scanner);
        let tokens = Preprocessor::new(parser, lexer, reporter.clone());
        Self::parse_tokens(reporter, parser.codemap.clone(), tokens)
    }

    fn parse_tokens<S: IntoIterator<Item = Preprocessed>>(
        reporter: Reporter,
        codemap: Arc<CodeMap>,
        tokens: S,
    ) -> Result<Self, ParserError> {
        let result = Self::Parser::new().parse(&reporter, &codemap, tokens);
        to_parse_result(reporter, result)
    }
}

fn to_parse_result<T>(reporter: Reporter, result: Result<T, ParseError>) -> Result<T, ParserError> {
    match result {
        Ok(ast) => {
            if reporter.is_failed() {
                return Err(ParserError::ShowDiagnostic {
                    diagnostic: Diagnostic::error()
                        .with_message("parsing failed, see diagnostics for details"),
                });
            }
            Ok(ast)
        }
        Err(lalrpop_util::ParseError::User { error }) => Err(error.into()),
        Err(err) => Err(ParserError::from(err).into()),
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use pretty_assertions::assert_eq;

    use super::*;
    use crate::ast::*;

    use firefly_diagnostics::*;
    use firefly_intern::{Ident, Symbol};

    use crate::preprocessor::PreprocessorError;

    fn fail_with(reporter: Reporter, codemap: &CodeMap, message: &'static str) -> ! {
        use term::termcolor::{ColorChoice, StandardStream};

        let config = term::Config::default();
        let mut out = StandardStream::stderr(ColorChoice::Always);
        for diagnostic in errors.iter_diagnostics() {
            term::emit(&mut out, &config, codemap, &diagnostic).unwrap();
        }
        panic!(message);
    }

    fn parse<T, S>(config: ParseConfig, codemap: Arc<CodeMap>, input: S) -> T
    where
        T: Parse<T, Config = ParseConfig, Error = ParserError>,
        S: AsRef<str>,
    {
        let mut errors = Reporter::new();
        let parser = Parser::new(config, codemap);
        match parser.parse_string::<T, S>(errors.clone(), input) {
            Ok(ast) => return ast,
            Err(_errs) => fail_with(errors, &parser.codemap, "parse failed"),
        }
    }

    fn parse_fail<T, S>(config: ParseConfig, codemap: Arc<CodeMap>, input: S) -> Reporter
    where
        T: Parse<T, Config = ParseConfig, Error = ParserError>,
        S: AsRef<str>,
    {
        let mut errors = Reporter::new();
        let parser = Parser::new(config, codemap);
        match parser.parse_string::<T, S>(errors, input) {
            Err(()) => errors,
            _ => panic!("expected parse to fail, but it succeeded!"),
        }
    }

    macro_rules! module {
        ($codemap:expr, $name:expr, $body:expr) => {{
            let mut errs = Reporter::new();
            let codemap = $codemap;
            let module = Module::new_with_forms(
                &mut errs,
                codemap.clone(),
                SourceSpan::UNKNOWN,
                $name,
                $body,
            );
            if errs.is_failed() {
                fail_with(errs, codemap, "failed to create expected module!");
            }
            module
        }};
    }

    #[test]
    fn parse_empty_module() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(config, codemap.clone(), "-module(foo).");
        let expected = module!(&codemap, ident!("foo"), vec![]);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_module_with_multi_clause_function() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

foo([], Acc) -> Acc;
foo([H|T], Acc) -> foo(T, [H|Acc]).
",
        );

        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(foo).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![nil!(), var!(Acc)],
                guards: vec![],
                body: vec![var!(Acc)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(foo).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![cons!(var!(H), var!(T)), var!(Acc)],
                guards: vec![],
                body: vec![apply!(foo, var!(T), (cons!(var!(H), var!(Acc))))],
                compiler_generated: false,
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!("foo")),
            arity: 2,
            clauses,
            spec: None,
        }));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_if_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
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
        clauses.push((
            ident_opt!(unless).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![atom!(false)],
                guards: vec![],
                body: vec![atom!(true)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(unless).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![atom!(true)],
                guards: vec![],
                body: vec![atom!(false)],
                compiler_generated: false,
            },
        ));
        clauses.push((
            ident_opt!(unless).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![var!(Value)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::If(If {
                    span: SourceSpan::UNKNOWN,
                    clauses: vec![
                        Clause::for_if(
                            SourceSpan::UNKNOWN,
                            vec![Guard {
                                span: SourceSpan::UNKNOWN,
                                conditions: vec![Expr::BinaryExpr(BinaryExpr {
                                    span: SourceSpan::UNKNOWN,
                                    lhs: Box::new(var!(Value)),
                                    op: BinaryOp::Equal,
                                    rhs: Box::new(int!(0.into())),
                                })],
                            }],
                            vec![atom!(true)],
                            false,
                        ),
                        Clause::for_if(
                            SourceSpan::UNKNOWN,
                            vec![Guard {
                                span: SourceSpan::UNKNOWN,
                                conditions: vec![var!(Value)],
                            }],
                            vec![atom!(false)],
                            false,
                        ),
                        Clause::for_if(
                            SourceSpan::UNKNOWN,
                            vec![Guard {
                                span: SourceSpan::UNKNOWN,
                                conditions: vec![atom!(else)],
                            }],
                            vec![atom!(true)],
                            false,
                        ),
                    ],
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(unless)),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_case_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
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
        clauses.push((
            ident_opt!(typeof).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![var!(Value)],
                guards: vec![],
                body: vec![Expr::Case(Case {
                    span: SourceSpan::UNKNOWN,
                    expr: Box::new(var!(Value)),
                    clauses: vec![
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![nil!()],
                            guards: vec![],
                            body: vec![atom!(nil)],
                            compiler_generated: false,
                        },
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![cons!(var!(_), var!(_))],
                            guards: vec![],
                            body: vec![atom!(list)],
                            compiler_generated: false,
                        },
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![var!(N)],
                            guards: vec![Guard {
                                span: SourceSpan::UNKNOWN,
                                conditions: vec![apply!(atom!(is_number), (var!(N)))],
                            }],
                            body: vec![var!(N)],
                            compiler_generated: false,
                        },
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![var!(_)],
                            guards: vec![],
                            body: vec![atom!(other)],
                            compiler_generated: false,
                        },
                    ],
                })],
                compiler_generated: false,
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(typeof)),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_receive_expressions() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
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
        clauses.push((
            ident_opt!(loop).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![var!(State), var!(Timeout)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::Receive(Receive {
                    span: SourceSpan::UNKNOWN,
                    clauses: Some(vec![
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![tuple!(var!(From), tuple!(var!(Ref), var!(Msg)))],
                            guards: vec![],
                            body: vec![
                                Expr::BinaryExpr(BinaryExpr {
                                    span: SourceSpan::UNKNOWN,
                                    lhs: Box::new(var!(From)),
                                    op: BinaryOp::Send,
                                    rhs: Box::new(tuple!(var!(Ref), atom!(ok))),
                                }),
                                apply!(
                                    SourceSpan::UNKNOWN,
                                    atom!(handle_info),
                                    (var!(Msg), var!(State))
                                ),
                            ],
                            compiler_generated: false,
                        },
                        Clause {
                            span: SourceSpan::UNKNOWN,
                            patterns: vec![var!(_)],
                            guards: vec![],
                            body: vec![apply!(
                                SourceSpan::UNKNOWN,
                                atom!(exit),
                                (apply!(
                                    SourceSpan::UNKNOWN,
                                    io_lib,
                                    format,
                                    (
                                        Expr::Literal(Literal::String(ident!(
                                            "unexpected message: ~p~n"
                                        ))),
                                        cons!(var!(Msg), nil!())
                                    )
                                ))
                            )],
                            compiler_generated: false,
                        },
                    ]),
                    after: Some(After {
                        span: SourceSpan::UNKNOWN,
                        timeout: Box::new(var!(Timeout)),
                        body: vec![atom!(timeout)],
                    }),
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(loop)),
            arity: 2,
            clauses,
            spec: None,
        }));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_if() {
        let codemap = Arc::new(CodeMap::new());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
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
        clauses.push((
            ident_opt!(env).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![],
                guards: vec![],
                body: vec![atom!(test)],
                compiler_generated: false,
            },
        ));
        let env_fun = NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(env)),
            arity: 0,
            clauses,
            spec: None,
        };
        body.push(TopLevel::Function(env_fun));

        let mut clauses = Vec::new();
        clauses.push((
            ident_opt!(system_version).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![],
                guards: vec![],
                body: vec![int!(21.into())],
                compiler_generated: false,
            },
        ));
        let system_version_fun = NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(system_version)),
            arity: 0,
            clauses,
            spec: None,
        };
        body.push(TopLevel::Function(system_version_fun));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_preprocessor_warning_error() {
        // NOTE: Warnings are not printed with cfg(test), as we
        // cannot control where they end up without refactoring to pass
        // a writer everywhere. You can change this for testing by
        // going to the Preprocessor and finding the line where we handle
        // the warning directive and toggle the config flag
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let mut errs = parse_fail::<Module, &str>(
            config,
            codemap.clone(),
            "-module(foo).
-warning(\"this is a compiler warning\").
-error(\"this is a compiler error\").
",
        );
        match errs.errors.pop() {
            Some(ErrorOrWarning::Error(ParserError::Preprocessor {
                source: PreprocessorError::CompilerError { .. },
            })) => (),
            Some(err) => panic!(
                "expected compiler error, but got a different error instead: {:?}",
                err
            ),
            None => panic!("expected compiler error, but didn't get any errors!"),
        }

        match errs.errors.pop() {
            Some(ErrorOrWarning::Warning(ParserError::Preprocessor {
                source: PreprocessorError::WarningDirective { .. },
            })) => (),
            Some(err) => panic!(
                "expected warning directive, but got a different error instead: {:?}",
                err
            ),
            None => panic!("expected compiler error, but didn't get any errors!"),
        }
    }

    #[test]
    fn parse_try() {
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let result: Module = parse(
            config,
            codemap.clone(),
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
        clauses.push((
            ident_opt!(example).map(Name::Atom),
            Clause {
                span: SourceSpan::UNKNOWN,
                patterns: vec![var!(File)],
                guards: vec![],
                compiler_generated: false,
                body: vec![Expr::Try(Try {
                    span: SourceSpan::UNKNOWN,
                    exprs: vec![apply!(atom!(read), (var!(File)))],
                    clauses: Some(vec![Clause {
                        span: SourceSpan::UNKNOWN,
                        patterns: vec![tuple!(atom!(ok), var!(Contents))],
                        guards: vec![],
                        body: vec![tuple!(atom!(ok), var!(Contents))],
                        compiler_generated: false,
                    }]),
                    catch_clauses: Some(vec![
                        TryClause {
                            span: SourceSpan::UNKNOWN,
                            kind: Name::Atom(ident!(error)),
                            error: tuple!(var!(Mod), var!(Code)),
                            trace: ident!(_),
                            guard: None,
                            body: vec![tuple!(
                                atom!(error),
                                apply!(
                                    SourceSpan::UNKNOWN,
                                    var!(Mod),
                                    atom!(format_error),
                                    (var!(Code))
                                )
                            )],
                        },
                        TryClause {
                            span: SourceSpan::UNKNOWN,
                            kind: Name::Atom(ident!(throw)),
                            error: var!(Reason),
                            trace: ident!(_),
                            guard: None,
                            body: vec![tuple!(atom!(error), var!(Reason))],
                        },
                    ]),
                    after: Some(vec![apply!(
                        SourceSpan::UNKNOWN,
                        atom!(close),
                        (var!(File))
                    )]),
                })],
            },
        ));
        let mut body = Vec::new();
        body.push(TopLevel::Function(NamedFunction {
            span: SourceSpan::UNKNOWN,
            name: Name::Atom(ident!(example)),
            arity: 1,
            clauses,
            spec: None,
        }));
        let expected = module!(&codemap, ident!(foo), body);
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_try2() {
        let codemap = Arc::new(CodeMap::default());
        let config = ParseConfig::default();
        let _result: Module = parse(
            config,
            codemap.clone(),
            "-module(foo).

example(File < 2) ->
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

exw(File) ->
    case File of
        File < 2 ->
            ok
    end.
",
        );
    }

    #[test]
    fn parse_numbers() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

foo(F) -> F-1+1/1*1.

bar() -> - 2.
",
        );
    }

    #[test]
    fn parse_spec() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

-spec bar() -> number.
bar() -> 2.
",
        );
    }

    #[test]
    fn parse_binary_spec_constant() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            "-module(foo).

-type txs_hash() :: <<_:(32 * 8)>>.
-type a() :: <<_:A * (12 * 8)>>.
",
        );
    }

    #[test]
    fn parse_multi_line_string() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-deprecated([{woo, 0,
"testing testing"
"testing testing"}]).
woo() -> 2.
"#,
        );
    }

    #[test]
    fn parse_multi_line_record() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-record(some_record,
        {
            woohoo
        }).
"#,
        );
    }

    #[test]
    fn parse_removed_attribute() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).

-removed([{foo, 0, "we don't support this anymore!"}]).
"#,
        );
    }

    #[test]
    fn parse_on_load() {
        let _result: Module = parse(
            ParseConfig::default(),
            Arc::new(CodeMap::new()),
            r#"-module(foo).
-on_load(bar/0).

bar() -> yay.
"#,
        );
    }

    #[test]
    fn parse_elixir_enum_erl() {
        use std::io::Read;

        let file = std::fs::File::open("../test_data/Elixir.Enum.erl");
        let mut string = String::new();
        file.unwrap().read_to_string(&mut string).unwrap();

        let codemap = Arc::new(CodeMap::new());
        let _result: Module = parse(ParseConfig::default(), codemap, &string);
    }
}
